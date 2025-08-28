# app/scripts/generate_news_openai.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any
from openai import OpenAI

from .memory_context import fetch_weighted_context, compress_to_capsule

SYSTEM_STYLE = (
    "Ты — редактор проекта «Россия, которая получается». Пишешь хорошие новости о России будущего. "
    "Тон: ясный, доброжелательный, интеллектуальный. Стиль — в духе лакановского психоанализа: "
    "намёки на символическое/воображаемое/реальное, метафоры разрыва и сцепления, субъекта и большого Другого. "
    "Допускаются лёгкие ироничные отсылки к работам Славоя Жижека и Смулянского (без прямых цитат и академической тяжеловесности). "
    "Цель: вселять спокойный оптимизм, показывать рост, восстановление связей, заботу и технологическое развитие."
)

USER_TMPL = (
    "Сгенерируй {n} новостей. Учитывай «капсулу памяти» ниже — усиливай темы с большим весом экспоненциально. "
    "Формат ответа строго JSON-массив объектов, каждый объект:\n"
    "{{\"title\":\"...\",\"text\":\"...\",\"section\":\"main|side|list\",\"slug\":\"опционально\",\"tags\":\"опционально\"}}\n\n"
    "Требования к каждой новости:\n"
    "- Заголовок до 120 символов, без кликбейта, с мягким философским акцентом.\n"
    "- Текст 600–1200 символов, связный, конкретный, без политики текущего дня и без негативных деталей.\n"
    "- section: максимум 1 \"main\", до 3 \"side\", остальные \"list\".\n"
    "- В умеренной форме допускай отсылки к Лакану/Жижеку/Смулянскому (как рамки мысли), без цитат.\n"
    "- Старайся поддерживать непрерывность нарратива: возвращай важные образы/темы из капсулы, не повторяя формулировки дословно.\n\n"
    "Капсула памяти:\n{capsule}\n\n"
    "Верни только JSON-массив, без комментариев."
)

def _parse_json_array(txt: str) -> List[Dict[str, Any]]:
    start, end = txt.find("["), txt.rfind("]")
    if start != -1 and end != -1:
        txt = txt[start:end+1]
    data = json.loads(txt)
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON array")
    out = []
    for it in data:
        title = (it.get("title") or "").strip()
        text  = (it.get("text") or "").strip()
        if not title or not text:
            continue
        section = (it.get("section") or "list").strip()
        if section not in {"main","side","list"}:
            section = "list"
        slug = (it.get("slug") or "").strip()
        tags = it.get("tags") or ""
        out.append({"title": title, "text": text, "section": section, "slug": slug, "tags": tags})
    return out

def run(n: int = 3, model: str | None = None, lam: float = 0.10, context_items: int = 60) -> List[Dict[str, Any]]:
    """
    n — сколько новостей сгенерировать (1..10)
    lam — коэффициент экспоненциального затухания памяти (больше — быстрее забываем)
    context_items — сколько прошлых статей учитывать
    """
    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    weighted = fetch_weighted_context(max_items=context_items, lam=lam)
    capsule  = compress_to_capsule(weighted, model=model) if weighted else ""

    user = USER_TMPL.format(n=max(1, min(10, int(n))), capsule=capsule)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":SYSTEM_STYLE},
            {"role":"user","content":user},
        ],
        temperature=0.7,
        max_tokens=2000
    )
    txt = resp.choices[0].message.content.strip()
    arts = _parse_json_array(txt)

    # Почистим секции: не более 1 main и до 3 side
    mains = [a for a in arts if a["section"] == "main"][:1]
    sides = [a for a in arts if a["section"] == "side"][:3]
    rest  = [a for a in arts if a not in mains and a not in sides]
    for a in rest:
        a["section"] = "list"
    return mains + sides + rest
