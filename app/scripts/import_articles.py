import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import text as sql_text
from app import create_app, db
from app.models import Article

_flask_app = create_app()
_db = db

def _s(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)

import re

def make_slug(title, text, section, idx):
    # Use title if available, else text, else section+idx
    base = title or text or f"{section}-{idx}"
    s = base.lower()
    s = re.sub(r"[^a-z0-9а-яё]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or f"article-{section}-{idx}"

def import_articles(articles: List[Dict[str, Any]]) -> int:
    total = 0
    with _flask_app.app_context():
        try:
            _db.session.execute(sql_text("SET client_encoding TO 'UTF8'"))
        except Exception:
            pass

        for idx, a in enumerate(articles):
            if not isinstance(a, dict):
                print(f"[warn] skipping non-dict item: {a!r}")
                continue
            title = _s(a.get("title"))
            text = _s(a.get("text"))
            slug = _s(a.get("slug"))
            section = _s(a.get("section") or "list")
            created_at = a.get("created_at") or datetime.utcnow()

            # If slug is missing or empty, generate one
            if not slug:
                slug = make_slug(title, text, section, idx)

            # Ensure slug is unique in DB
            exists = Article.query.filter_by(slug=slug).first()
            if exists:
                slug = f"{slug}-{int(datetime.utcnow().timestamp())}"

            known = {"title", "text", "slug", "section", "tags", "created_at"}
            extra = {k: v for k, v in a.items() if k not in known}
            tags = _s(a.get("tags"))
            if extra:
                tags = (tags + "\n" if tags else "") + json.dumps(extra, ensure_ascii=False)

            rec = Article(
                title=title,
                text=text,
                tags=tags,
                slug=slug,
                section=section,
                created_at=created_at,
            )
            _db.session.add(rec)
            total += 1
        _db.session.commit()
        return total
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_articles.py <path_to_json>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Flatten the structure
    articles = []
    if isinstance(raw, dict):
        for section, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item["section"] = section
                articles.append(item)
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        entry = dict(entry)
                        entry["section"] = section
                        articles.append(entry)
    else:
        articles = raw
    count = import_articles(articles)
    print(f"Imported {count} articles.")