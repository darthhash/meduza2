# scripts/import_articles.py
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import text as sql_text

def _s(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)

def import_articles(articles: List[Dict[str, Any]]) -> int:
    """
    Импортирует список статей в таблицу Article (или через сырой SQL),
    принудительно включая UTF-8.
    Ожидает поля: title, text, tags, slug, section, created_at.
    """
    from app import app as _flask_app
    from app import db as _db
    # Пытаемся использовать модель, если она есть
    Article = None
    try:
        from app import Article as _Article  # type: ignore
        Article = _Article
    except Exception:
        Article = None

    total = 0
    with _flask_app.app_context():
        # 1) клиентская кодировка на всякий случай
        try:
            _db.session.execute(sql_text("SET client_encoding TO 'UTF8'"))
        except Exception:
            pass

        if Article is not None:
            # ORM-путь
            for a in articles:
                rec = Article(
                    title=_s(a.get("title")),
                    text=_s(a.get("text")),
                    tags=_s(a.get("tags")),
                    slug=_s(a.get("slug")),
                    section=_s(a.get("section") or "list"),
                    created_at=a.get("created_at") or datetime.utcnow(),
                )
                _db.session.add(rec)
                total += 1
            _db.session.commit()
            return total

        # 2) Сырой SQL (если модели нет)
        # Подставь реальное имя таблицы, если у тебя другое:
        table = "articles"
        for a in articles:
            _db.session.execute(
                sql_text(
                    f"""INSERT INTO {table}
                        (title, text, tags, slug, section, created_at)
                        VALUES (:title, :text, :tags, :slug, :section, :created_at)"""
                ),
                {
                    "title": _s(a.get("title")),
                    "text": _s(a.get("text")),
                    "tags": _s(a.get("tags")),
                    "slug": _s(a.get("slug")),
                    "section": _s(a.get("section") or "list"),
                    "created_at": a.get("created_at") or datetime.utcnow(),
                },
            )
            total += 1
        _db.session.commit()
        return total
