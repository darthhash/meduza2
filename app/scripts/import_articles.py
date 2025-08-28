# app/scripts/import_articles.py (добавь/обнови)
import os, sys, json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import text as sql_text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app import create_app, db
from app.models import Article
from slugify import slugify as _slugify

_flask_app = create_app()

def _slugify_unique(base: str) -> str:
    slug = _slugify(base or "", lowercase=True, max_length=140)
    if not slug:
        slug = f"art-{int(datetime.utcnow().timestamp())}"
    orig = slug
    i = 2
    with _flask_app.app_context():
        while Article.query.filter_by(slug=slug).first():
            slug = f"{orig}-{i}"
            i += 1
    return slug

def import_articles_from_list(items: List[Dict[str, Any]]) -> List[int]:
    """Принимает список {"title","text","section","slug","tags","created_at"}"""
    ids = []
    with _flask_app.app_context():
        for a in items:
            title   = (a.get("title") or "").strip()
            text    = (a.get("text") or "").strip()
            section = (a.get("section") or "list").strip()
            slug    = (a.get("slug") or "").strip()
            tags    = a.get("tags") or ""
            created = a.get("created_at")

            if not title or not text:
                continue

            if not slug:
                slug = _slugify_unique(title)
            else:
                slug = _slugify_unique(slug)  # и так обеспечим уникальность

            if created:
                try:
                    # Принимаем ISO/строку секунд — приводим к datetime
                    if isinstance(created, (int, float)):
                        created = datetime.utcfromtimestamp(int(created))
                    elif isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z","").strip())
                except Exception:
                    created = None

            art = Article(
                title=title, text=text, section=section, slug=slug,
                tags=(tags if isinstance(tags, str) else json.dumps(tags, ensure_ascii=False)),
                created_at=created or datetime.utcnow()
            )
            db.session.add(art)
            db.session.flush()
            ids.append(art.id)

        db.session.commit()
    return ids
