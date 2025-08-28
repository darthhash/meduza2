# app/scripts/memory_context.py
from __future__ import annotations
import math
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import desc
from openai import OpenAI

from app import db
from app.models import Article

def _weight(i: int, lam: float) -> float:
    # i = 0 для самой свежей; ламбда 0.08..0.15 — нормальный диапазон
    return math.exp(-lam * i)

def fetch_weighted_context(max_items: int = 60, lam: float = 0.10) -> List[Dict[str, Any]]:
    q = Article.query.order_by(desc(Article.created_at)).limit(max_items)
    rows = q.all()
    items = []
    for i, a in enumerate(rows):
        items.append({
            "title": a.title,
            "section": a.section or "list",
            "tags": a.tags or "",
            "created_at": a.created_at or datetime.utcnow(),
            "weight": _weight(i, lam),
        })
    return items

def compress_to_capsule(items: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    """
    Превращаем взвешенный список в короткую «капсулу памяти»: 8–12 пунктов
    с акцентом на более высокие веса.
    """
    if not items:
        return ""
    # Готовим короткий сырой контент для модели
    lines = []
    for it in items:
        line = f"- ({it['weight']:.3f}) [{it['section']}] {it['title']}"
        if it.get("tags"):
            line += f" | tags: {str(it['tags'])[:120]}"
        lines.append(line)

    client = OpenAI()
    sys = (
        "Ты — редактор-куратор памяти. Получишь список прошлых новостей с весами. "
        "Сожми их в 8–12 маркеров тем/мотивов/образов. Чем выше вес — тем сильнее акцент. "
        "Не делай выводов из внешней реальности; работай только с данным списком."
    )
    usr = "Список прошлых новостей с весами важности:\n" + "\n".join(lines)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()
