# scripts/generate_news.py
# -*- coding: utf-8 -*-
"""
Генерирует НОВЫЕ ВЫМЫШЛЕННЫЕ новости о России ближайшего будущего
на основании последних статей из БД с экспоненциальным затуханием «в прошлое».
Подача: интерпретации в духе Лакана / Жижека / Смулянского.

Что делает:
- Тянет последние K статей из БД (SQLAlchemy-модель Article или сырым SQL).
- Строит контекст (новые важнее) и извлекает темы из заголовков/тегов.
- Генерит N новых статей (JSON: title, section, tags, text).
- Записывает в scripts/articles_payload.py и опционально импортит в БД.

ENV (локально через .env, на Railway — Variables):
  LLM_BACKEND=llama|transformers
  GGUF_REPO_ID=TheBloke/Qwen2.5-7B-Instruct-GGUF
  GGUF_FILENAME=qwen2.5-7b-instruct.Q4_K_M.gguf
  HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
  MAX_TOKENS=1024
  TEMPERATURE=0.7

  # История/контекст:
  LAST_K=40
  HALF_LIFE=10
  CTX_MAX_CHARS=8000

  # Hugging Face auth:
  HF_TOKEN=hf_...                 # или HUGGINGFACE_HUB_TOKEN=hf_...

  # PROMPT overrides (опционально):
  PROMPT_SYSTEM / PROMPT_USER  или  PROMPT_MODULE + PROMPT_SYSTEM_VAR/PROMPT_USER_VAR
"""

import os, sys, re, json, argparse, textwrap, html
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

# ─── .env loader (локально) ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv
    _env_path = find_dotenv(usecwd=True)
    if _env_path:
        load_dotenv(_env_path)
except Exception:
    pass

# ─── импорт из проекта / slugify ───────────────────────────────────────────
sys.path.insert(0, os.path.abspath("."))

def _wire_slugify():
    # 1) попробовать забрать slugify из проекта
    for path in ("app", "app.utils", "app.helpers", "app.lib", "app.core", "app.common"):
        try:
            mod = __import__(path, fromlist=["*"])
            fn = getattr(mod, "slugify", None)
            if callable(fn): return fn
        except Exception:
            pass
    # 2) внешний пакет python-slugify
    try:
        from slugify import slugify as _ext_slugify
        return _ext_slugify
    except Exception:
        pass
    # 3) фолбэк (русский транслит)
    RU = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"i",
          "к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f",
          "х":"h","ц":"c","ч":"ch","ш":"sh","щ":"sch","ы":"y","э":"e","ю":"yu","я":"ya","ь":"","ъ":""}
    def _fallback_slugify(s: str) -> str:
        s = (s or "").strip().lower()
        t = "".join(RU.get(ch, ch) for ch in s)
        t = re.sub(r"[^a-z0-9]+", "-", t)
        t = re.sub(r"-{2,}", "-", t).strip("-")
        return t
    return _fallback_slugify
 #defw
slugify = _wire_slugify()

# ───────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ───────────────────────────────────────────────────────────────────────────

def getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default

def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)

def ts_from_iso(s: Optional[str]) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

# ───────────────────────────────────────────────────────────────────────────
# ЧТЕНИЕ ИЗ БД
# ───────────────────────────────────────────────────────────────────────────

def fetch_recent_articles_from_db(limit: int = 40) -> List[Dict[str, Any]]:
    """
    Возвращает [{title, text, tags, slug, section, created_at}, ...]
    """
    # 1) SQLAlchemy-модель Article из app.py (через пакет app)
    try:
        from app import app as _flask_app  # type: ignore
        from app import db as _db         # type: ignore
        from app import Article as _Article  # type: ignore

        if _Article is not None and _flask_app is not None:
            with _flask_app.app_context():
                q = (
                    _Article.query
                    .order_by(_Article.created_at.desc() if hasattr(_Article, "created_at")
                              else _Article.id.desc())
                    .limit(limit)
                )
                rows = q.all()
                out = []
                for r in rows:
                    out.append({
                        "title": getattr(r, "title", "") or "",
                        "text": getattr(r, "text", "") or "",
                        "tags": getattr(r, "tags", "") or "",
                        "slug": getattr(r, "slug", "") or "",
                        "section": getattr(r, "section", "") or "list",
                        "created_at": (getattr(r, "created_at", None) or "").isoformat()
                                      if hasattr(r, "created_at") and getattr(r, "created_at") else ""
                    })
                out.sort(key=lambda x: ts_from_iso(x["created_at"]), reverse=True)
                return out
    except Exception as e:
        print("[warn] SQLAlchemy Article path failed:", e)

    # 2) Сырой SQL
    try:
        from app import app as _flask_app  # type: ignore
        from app import db as _db          # type: ignore
        from sqlalchemy import text as sql_text  # type: ignore

        candidate_tables = ["articles", "news", "posts"]
        candidate_cols = [
            ("title", "text", "tags", "slug", "section", "created_at"),
            ("title", "body", "tags", "slug", "section", "created_at"),
            ("title", "content", "tags", "slug", "section", "created_at"),
        ]

        if _flask_app is None or _db is None:
            raise RuntimeError("Flask app/db not available for raw SQL")

        with _flask_app.app_context():
            for tbl in candidate_tables:
                for cols in candidate_cols:
                    cols_sql = ", ".join(cols)
                    q = f"SELECT {cols_sql} FROM {tbl} ORDER BY created_at DESC LIMIT :lim"
                    try:
                        rows = _db.session.execute(sql_text(q), {"lim": limit}).fetchall()
                        if not rows:
                            continue
                        out = []
                        for row in rows:
                            d = dict(zip(cols, row))
                            out.append({
                                "title": d.get("title") or "",
                                "text": d.get("text") or d.get("body") or d.get("content") or "",
                                "tags": d.get("tags") or "",
                                "slug": d.get("slug") or "",
                                "section": d.get("section") or "list",
                                "created_at": d.get("created_at").isoformat()
                                              if hasattr(d.get("created_at"), "isoformat") and d.get("created_at")
                                              else (d.get("created_at") or "")
                            })
                        out.sort(key=lambda x: ts_from_iso(x["created_at"]), reverse=True)
                        return out
                    except Exception:
                        continue
    except Exception as e:
        print("[warn] raw SQL path failed:", e)

    return []

# ───────────────────────────────────────────────────────────────────────────
# КОНТЕКСТ С ЭКСПОНЕНЦИАЛЬНЫМ ЗАТУХАНИЕМ
# ───────────────────────────────────────────────────────────────────────────

RU_STOP = set("""
и в во что на для по как не от из у к до о над под при про без между или но либо либоже
это этой этот эта эти тех там тут такой такая такие было были был была будет будут
""".split())

def tokenize_ru(s: str) -> List[str]:
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.I | re.U)
    s = s.replace("_", " ")
    toks = [t.lower() for t in s.split() if len(t) >= 4 and t.lower() not in RU_STOP]
    return toks

def exp_weights(n: int, half_life: int) -> List[float]:
    return [0.5 ** (i / max(1, half_life)) for i in range(n)]

def build_context(arts: List[Dict[str, Any]], last_k: int, half_life: int, max_chars: int) -> str:
    subset = arts[:max(1, last_k)]
    weights = exp_weights(len(subset), half_life)
    chunks, total = [], 0

    for i, a in enumerate(subset):
        w = weights[i]
        dt = a.get("created_at") or ""
        ds = ""
        if dt:
            try:
                ds = datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            except Exception:
                ds = ""
        title = (a.get("title") or "").strip()
        tags = (a.get("tags") or "").strip()
        plain = strip_html(a.get("text") or "")
        block_len = int(400 * (1 + 3 * w))  # 400..1600 (новые — длиннее)
        brief = plain[:block_len].strip()

        head = f"- ({ds}) {title}"
        if tags:
            head += f" — теги: {tags}"
        piece = f"{head}\n{brief}\n"
        if total + len(piece) <= max_chars:
            chunks.append(piece)
            total += len(piece)
        else:
            break

    return ("Предыдущие публикации (новые → старые):\n" + "\n".join(chunks)).strip()

def derive_topics(arts: List[Dict[str, Any]], n: int, last_k: int, half_life: int) -> List[str]:
    subset = arts[:max(1, last_k)]
    weights = exp_weights(len(subset), half_life)
    bag = Counter()

    for i, a in enumerate(subset):
        w = weights[i]
        tags = a.get("tags") or ""
        title = a.get("title") or ""
        for t in re.split(r"[,\|/;]+", tags):
            t = t.strip()
            if len(t) >= 3:
                bag[t] += 1.5 * w
        for tok in tokenize_ru(title):
            bag[tok] += 1.0 * w

    if not bag:
        return ["общество будущего", "технологии будущего", "политэкономия будущего"][:n]

    topics = []
    for word, _score in bag.most_common(60):
        topic = word
        if not any(topic.lower() == t.lower() for t in topics):
            topics.append(topic)
        if len(topics) >= n:
            break
    return topics[:n] if topics else ["будущее России"] * n

# ───────────────────────────────────────────────────────────────────────────
# LLM BACKENDS
# ───────────────────────────────────────────────────────────────────────────

class LlamaCppBackend:
    """
    Скачиваем снапшот репозитория (только *.gguf), выбираем файл по подстроке из ENV
    или самый крупный .gguf. Передаём token, если задан (HF_TOKEN/HUGGINGFACE_HUB_TOKEN).
    """
    
    def __init__(self, repo_id: str, filename: str, max_tokens: int = 1024, temperature: float = 0.7):
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
        from llama_cpp import Llama

        self.max_tokens = max_tokens
        self.temperature = temperature

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        print(f"[llama] resolve GGUF: repo='{repo_id}', filename='{filename}', token={'set' if token else 'none'}")

        try:
            want = (filename or "").strip()
            patterns = [want, want.lower(), want.upper(), f"*{want}*"] if want else ["*.gguf"]
            snap_dir = snapshot_download(repo_id=repo_id, allow_patterns=patterns, token=token, local_dir="data/models")

        except RepositoryNotFoundError as e:
            raise SystemExit(
                f"\n[ERR] HF repo not found or gated: '{repo_id}'. "
                f"Проверь имя репозитория и что ты залогинен.\n"
                f"Лечится так: 1) создай токен на huggingface.co/settings/tokens, "
                f"2) положи его в .env как HF_TOKEN=hf_xxx, "
                f"3) (если нужно) нажми Accept на странице модели.\n{e}"
            )
        except GatedRepoError as e:
            raise SystemExit(
                f"\n[ERR] Repo is gated: '{repo_id}'. Нажми 'Access'/'Accept' на странице модели "
                f"и/или укажи валидный HF_TOKEN в .env.\n{e}"
            )
        except Exception as e:
            raise SystemExit(
                f"\n[ERR] snapshot_download failed. Если видишь 401 — добавь HF_TOKEN в .env.\n{e}"
            )

        ggufs = sorted(Path(snap_dir).rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        single = [p for p in ggufs if "-of-" not in p.name.lower()]  # исключить шардированные куски
        ggufs = single or ggufs

        if not ggufs:
            raise SystemExit(f"[ERR] В снапшоте нет .gguf файлов: {repo_id}")

        chosen = None
        fn_l = (filename or "").lower().strip()
        if fn_l:
            for p in ggufs:
                if fn_l in p.name.lower():
                    chosen = p
                    break
        if chosen is None:
            chosen = ggufs[0]  # самый большой — обычно цельный

        model_path = str(chosen)
        print(f"[llama] using model: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=min(8, os.cpu_count() or 8),
        )

    def chat(self, system: str, user: str) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()

class TransformersBackend:
    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.7):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        self.pipe = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            ),
            max_new_tokens=max_tokens,
            do_sample=True, temperature=temperature, top_p=0.9, repetition_penalty=1.05,
        )
        self.max_tokens, self.temperature = max_tokens, temperature

    def chat(self, system: str, user: str) -> str:
        prompt = f"<|system|>\n{system}\n</|system|>\n<|user|>\n{user}\n</|user|>\n<|assistant|>\n"
        out = self.pipe(prompt)[0]["generated_text"]
        m = re.search(r"<\|assistant\|>\n(.+)", out, re.S)
        return (m.group(1).strip() if m else out.strip())

# ───────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ (дефолт) + переопределения из ENV/модуля
# ───────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "Ты — редактор отдела «Россия будущего». Все новости ВЫМЫШЛЕННЫЕ. "
    "Отталкивайся от контекстного дайджеста прошлых публикаций, где новые важнее старых. "
    "Подавай материал аналитически: через оптику психоанализа и критической теории "
    "(Лакан, Жижек, Смулянский) — но без цитат и именованных ссылок. "
    "Не давай проверяемых фактов, реальных дат, ФИО. "
    "Вывод строго в JSON:\n"
    '{ "title": "...<=120...", "section": "main|list", "tags": "теги,через,запятую", "text": "<p>...HTML...</p>" } '
    "— без пролога/эпилога/подписей."
)

DEFAULT_USER_TMPL = (
    "Контекст (новые → старые, экспоненциальное затухание):\n{context}\n\n"
    "Сгенерируй ВЫМЫШЛЕННУЮ новость о России ближайшего будущего на тему: «{topic}».\n"
    "Требования:\n"
    "- 2–5 абзацев по 400–800 символов, HTML (<p>...</p>, можно 1–2 <h3>). Без изображений и таблиц.\n"
    "- Объясни «что произошло» через конструкции желаний/отсутствия/символического порядка (Лакан), "
    "идеологического интерпеллирования/события (Жижек), микроаналитики смысла (Смулянский) — "
    "без прямых цитат и фамилий в тексте.\n"
    "- Без реальных дат/фамилий/цифр, без кликбейта.\n"
    "- Верни СТРОГО один JSON-объект согласно формату выше."
)

def load_prompts() -> tuple[str, str]:
    module_name = os.getenv("PROMPT_MODULE")
    sys_var = os.getenv("PROMPT_SYSTEM_VAR", "SYSTEM_PROMPT")
    usr_var = os.getenv("PROMPT_USER_VAR", "USER_PROMPT")
    if module_name:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            s = getattr(mod, sys_var)
            u = getattr(mod, usr_var)
            return s, u
        except Exception as e:
            print("[warn] PROMPT_MODULE load failed:", e)
    s_env, u_env = os.getenv("PROMPT_SYSTEM"), os.getenv("PROMPT_USER")
    if (s_env or "").strip() and (u_env or "").strip():
        return s_env, u_env
    return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TMPL

SYSTEM_PROMPT, USER_TMPL = load_prompts()

# ───────────────────────────────────────────────────────────────────────────
# ГЕНЕРАЦИЯ СТАТЬИ
# ───────────────────────────────────────────────────────────────────────────

def parse_json_or_fallback(raw: str, topic: str) -> Dict[str, Any]:
    try:
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        data = json.loads(s)
    except Exception:
        title = (raw.split("\n", 1)[0] or topic).strip()[:120]
        body = "<p>" + re.sub(r"\n{2,}", "</p><p>", raw).strip() + "</p>"
        data = {"title": title, "section": "list", "tags": topic, "text": body}
    data["title"] = (data.get("title") or topic)[:120]
    data["section"] = (data.get("section") or "list") if (data.get("section") in ("main","list","side")) else "list"
    text = (data.get("text") or "").strip()
    if not text.startswith("<"):
        text = "<p>" + text.replace("\n\n","</p><p>").replace("\n"," ") + "</p>"
    data["text"] = text
    return data

def generate_one(llm, topic: str, context: str) -> Dict[str, Any]:
    user_prompt = USER_TMPL.format(topic=topic, context=context)
    raw = llm.chat(SYSTEM_PROMPT, user_prompt)
    data = parse_json_or_fallback(raw, topic)
    data["slug"] = slugify(data["title"])
    data["created_at"] = datetime.utcnow().isoformat()
    extra_tags = "Лакан,Жижек,Смулянский,психоанализ,идеология"
    data["tags"] = (data.get("tags") or topic)
    if extra_tags:
        seen, merged = set(), []
        for t in (str(data["tags"]) + "," + extra_tags).split(","):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower()); merged.append(t)
        data["tags"] = ",".join(merged)
    return data

def write_payload(articles: List[Dict[str, Any]], path: str = "scripts/articles_payload.py"):
    lines = ["ARTICLES = [\n"]
    for a in articles:
        text = a["text"].replace("\\", "\\\\").replace('"', '\\"')
        title = a["title"].replace("\\", "\\\\").replace('"', '\\"')
        tags = (a.get("tags") or "").replace("\\", "\\\\").replace('"', '\\"')
        lines.append("  {\n")
        lines.append(f'    "title": "{title}",\n')
        lines.append(f'    "slug": "{a["slug"]}",\n')
        lines.append(f'    "section": "{a["section"]}",\n')
        if tags:
            lines.append(f'    "tags": "{tags}",\n')
        lines.append(f'    "created_at": "{a["created_at"]}",\n')
        lines.append('    "text": (\n')
        for part in textwrap.wrap(text, width=120, break_long_words=False, break_on_hyphens=False):
            lines.append(f'      "{part}"\n')
        lines.append("    )\n")
        lines.append("  },\n")
    lines.append("]\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"[ok] wrote {path}")

# ───────────────────────────────────────────────────────────────────────────
# Импорт в БД: поддержка import_articles() и import_articles(articles)
# ───────────────────────────────────────────────────────────────────────────

def do_import_articles(articles: List[Dict[str, Any]]):
    try:
        import importlib, inspect
        imp_mod = importlib.import_module("scripts.import_articles")
        try:
            importlib.reload(imp_mod)
        except Exception:
            pass
        fn = getattr(imp_mod, "import_articles", None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) >= 1:
                    return fn(articles)
                else:
                    setattr(imp_mod, "ARTICLES", articles)
                    return fn()
            except Exception:
                setattr(imp_mod, "ARTICLES", articles)
                return fn()
        else:
            print("[warn] scripts.import_articles: function import_articles not found")
    except Exception as e:
        print("[warn] import into DB failed:", e)

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="сколько новых статей сгенерировать")
    parser.add_argument("--last-k", type=int, default=getenv_int("LAST_K", 40))
    parser.add_argument("--half-life", type=int, default=getenv_int("HALF_LIFE", 10))
    parser.add_argument("--ctx-max-chars", type=int, default=getenv_int("CTX_MAX_CHARS", 8000))
    parser.add_argument("--import", dest="do_import", action="store_true", help="сразу импортировать в БД")
    args = parser.parse_args()

    # 1) история
    history = fetch_recent_articles_from_db(limit=args.last_k)
    if not history:
        print("[warn] нет статей в БД → контекст пустой (сгенерим без истории)")

    # 2) контекст и темы
    context = build_context(history, last_k=args.last_k, half_life=args.half_life, max_chars=args.ctx_max_chars)
    topics = derive_topics(history, n=args.n, last_k=args.last_k, half_life=args.half_life)

    # 3) модель
    backend = getenv_str("LLM_BACKEND", "llama").lower()
    max_tokens = getenv_int("MAX_TOKENS", 1024)
    temperature = float(getenv_str("TEMPERATURE", "0.7"))

    if backend == "llama":
        repo = getenv_str("GGUF_REPO_ID", "TheBloke/Qwen2.5-7B-Instruct-GGUF")
        fname = getenv_str("GGUF_FILENAME", "qwen2.5-7b-instruct.Q4_K_M.gguf")
        llm = LlamaCppBackend(repo, fname, max_tokens=max_tokens, temperature=temperature)
    else:
        model_id = getenv_str("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        llm = TransformersBackend(model_id, max_tokens=max_tokens, temperature=temperature)

    # 4) генерация
    articles = []
    for i, t in enumerate(topics):
        art = generate_one(llm, t, context)
        art["section"] = "main" if i == 0 else "list"
        articles.append(art)

    # 5) запись + импорт
    write_payload(articles, path="scripts/articles_payload.py")
    if args.do_import:
        do_import_articles(articles)
        print(f"[ok] imported {len(articles)} articles into DB")

if __name__ == "__main__":
    main()
