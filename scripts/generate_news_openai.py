# scripts/generate_news_openai.py
# -*- coding: utf-8 -*-

"""
Самодостаточный генератор вымышленных новостей:
- тянет последние статьи из БД (ORM Article или сырой SQL),
- строит контекст с экспоненциальным затуханием,
- извлекает темы,
- генерит JSON-статьи через OpenAI Chat (модель по ENV, по умолчанию gpt-4o-mini),
- добавляет изображение (OpenAI Images или placeholder),
- пишет scripts/articles_payload.py,
- по желанию импортирует в БД (scripts.import_articles.import_articles),
- экспортирует функцию run(...) для /newsgen/run.
"""

from __future__ import annotations
import os, re, json, textwrap, math, html, base64, io, pathlib, random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests
import urllib.parse

# ───────────────────────────────────────────────────────────────────────────
# .env (локально полезно; на Railway можно не нужно)
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _envp = find_dotenv(usecwd=True)
    if _envp:
        load_dotenv(_envp)
except Exception:
    pass

# ───────────────────────────────────────────────────────────────────────────
# Утилиты и безопасный slugify
def _wire_slugify():
    # 1) попробовать slugify из твоего проекта (разные места)
    for path in ("app", "app.utils", "app.helpers", "app.lib", "app.core", "app.common"):
        try:
            mod = __import__(path, fromlist=["*"])
            if hasattr(mod, "slugify") and callable(getattr(mod, "slugify")):
                return getattr(mod, "slugify")
        except Exception:
            pass
    # 2) python-slugify
    try:
        from slugify import slugify as _ext  # type: ignore
        return _ext
    except Exception:
        pass
    # 3) простой фолбэк для кириллицы
    RU = {
        "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"i",
        "к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f",
        "х":"h","ц":"c","ч":"ch","ш":"sh","щ":"sch","ы":"y","э":"e","ю":"yu","я":"ya","ь":"","ъ":""
    }
    def _fallback(s: str) -> str:
        s = (s or "").strip().lower()
        t = "".join(RU.get(ch, ch) for ch in s)
        t = re.sub(r"[^a-z0-9]+", "-", t)
        t = re.sub(r"-{2,}", "-", t).strip("-")
        return t
    return _fallback

slugify = _wire_slugify()

def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)

def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

def ts_from_iso(s: Optional[str]) -> float:
    if not s: return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

# ───────────────────────────────────────────────────────────────────────────
# Чтение последних статей из БД
def fetch_recent_articles_from_db(limit: int = 40) -> List[Dict[str, Any]]:
    """
    Возвращает список словарей: {title, text, tags, slug, section, created_at} (новые → старые).
    1) Пытается взять ORM Article из пакета app.
    2) Фолбэк — сырой SQL для типовых таблиц/колонок.
    """
    # ORM
    try:
        from app import app as _flask_app  # type: ignore
        from app import db as _db          # type: ignore
        from app import Article as _Article  # type: ignore
        if _flask_app is not None and _Article is not None:
            with _flask_app.app_context():
                order_col = _Article.created_at if hasattr(_Article, "created_at") else _Article.id
                rows = _Article.query.order_by(order_col.desc()).limit(limit).all()
                out = []
                for r in rows:
                    created = ""
                    try:
                        v = getattr(r, "created_at", None)
                        created = v.isoformat() if v else ""
                    except Exception:
                        pass
                    out.append({
                        "title":   getattr(r, "title", "") or "",
                        "text":    getattr(r, "text", "") or "",
                        "tags":    getattr(r, "tags", "") or "",
                        "slug":    getattr(r, "slug", "") or "",
                        "section": getattr(r, "section", "") or "list",
                        "created_at": created,
                    })
                out.sort(key=lambda x: ts_from_iso(x["created_at"]), reverse=True)
                return out
    except Exception as e:
        print("[warn] SQLAlchemy Article path failed:", e)

    # Сырой SQL
    try:
        from app import app as _flask_app  # type: ignore
        from app import db as _db          # type: ignore
        from sqlalchemy import text as sql_text  # type: ignore
        if _flask_app is None or _db is None:
            raise RuntimeError("Flask app/db not available")
        candidate_tables = ["articles", "news", "posts"]
        candidate_cols = [
            ("title","text","tags","slug","section","created_at"),
            ("title","body","tags","slug","section","created_at"),
            ("title","content","tags","slug","section","created_at"),
        ]
        with _flask_app.app_context():
            for tbl in candidate_tables:
                for cols in candidate_cols:
                    cols_sql = ", ".join(cols)
                    try:
                        res = _db.session.execute(
                            sql_text(f"SELECT {cols_sql} FROM {tbl} ORDER BY created_at DESC LIMIT :lim"),
                            {"lim": limit}
                        )
                        rows = res.fetchall()
                        if not rows:
                            continue
                        out = []
                        for row in rows:
                            d = dict(zip(cols, row))
                            created = d.get("created_at")
                            if hasattr(created, "isoformat"):
                                created = created.isoformat()
                            created = created or ""
                            out.append({
                                "title":   (d.get("title") or "")[:255],
                                "text":    d.get("text") or d.get("body") or d.get("content") or "",
                                "tags":    d.get("tags") or "",
                                "slug":    d.get("slug") or "",
                                "section": d.get("section") or "list",
                                "created_at": created,
                            })
                        out.sort(key=lambda x: ts_from_iso(x["created_at"]), reverse=True)
                        return out
                    except Exception:
                        continue
    except Exception as e:
        print("[warn] raw SQL path failed:", e)
    return []

# ───────────────────────────────────────────────────────────────────────────
# Контекст с экспоненциальным затуханием + темы
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
        block_len = int(400 * (1 + 3 * w))  # 400..1600
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

from collections import Counter
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
        if not any(word.lower() == t.lower() for t in topics):
            topics.append(word)
        if len(topics) >= n:
            break
    return topics[:n] if topics else ["будущее России"] * n

# ───────────────────────────────────────────────────────────────────────────
# Промпты
DEFAULT_SYSTEM_PROMPT = (
    "Ты — редактор отдела «Россия будущего». Все новости ВЫМЫШЛЕННЫЕ. "
    "Отталкивайся от дайджеста прошлых публикаций (новые важнее старых). "
    "Подавай через оптику психоанализа/критической теории (Лакан, Жижек, Смулянский) — без цитат/имён. "
    "Не давай реальных дат/ФИО. Вывод строго JSON одного объекта:\n"
    '{ "title": "...<=120...", "section": "main|list", "tags": "теги,через,запятую", "text": "<p>...HTML...</p>" }'
)

DEFAULT_USER_TMPL = (
    "Контекст (новые → старые, экспоненциальное затухание):\n{context}\n\n"
    "Сгенерируй ВЫМЫШЛЕННУЮ новость о России ближайшего будущего на тему: «{topic}».\n"
    "Требования:\n"
    "- 2–5 абзацев по 400–800 символов, HTML (<p>...</p>, можно 1–2 <h3>), без таблиц.\n"
    "- Объясни «что произошло» через желания/отсутствие/символический порядок (Лакан), "
    "событие/интерпелляцию (Жижек), микроаналитику смысла (Смулянский) — без фамилий/цитат.\n"
    "- Без реальных дат/фамилий/цифр, без кликбейта.\n"
    "- Верни строго один JSON-объект указанного формата."
)

SYSTEM_PROMPT = os.getenv("PROMPT_SYSTEM", DEFAULT_SYSTEM_PROMPT)
USER_TMPL = os.getenv("PROMPT_USER", DEFAULT_USER_TMPL)

# ───────────────────────────────────────────────────────────────────────────
# OpenAI Chat (дешёвая модель) + санитизация ключа
import re as _re

def _clean_openai_env_nonascii() -> List[Tuple[str,str]]:
    bad = []
    for key in ("OPENAI_PROJECT", "OPENAI_ORGANIZATION", "OPENAI_ORG_ID", "OPENAI_ORG"):
        val = os.getenv(key)
        if val and not val.isascii():
            bad.append((key, val))
            os.environ.pop(key, None)
    return bad

def _sanitize_api_key(k: str) -> str:
    if not k: return k
    k = k.strip()
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":  # длинные тире → "-"
        k = k.replace(ch, "-")
    k = k.replace("\u00A0","").replace(" ", "")             # NBSP/пробелы
    for ch in ('“','”','„','«','»',"'",'"','`'):            # кавычки
        k = k.replace(ch, "")
    k = _re.sub(r'(?:X?0D0A|X?0A|X?0D)$', '', k, flags=_re.IGNORECASE)  # хвосты копипасты
    if not k.isascii():
        bad = ''.join(sorted(set(c for c in k if not c.isascii())))
        raise ValueError(f"OPENAI_API_KEY contains non-ASCII characters: {repr(bad)}")
    if not (k.startswith("sk-") or k.startswith("sk_prov-") or k.startswith("sk-proj-")):
        raise ValueError("OPENAI_API_KEY looks invalid (expected to start with 'sk-').")
    return k

class OpenAIChat:
    def __init__(self, model: str, max_tokens: int = 900, temperature: float = 0.7):
        raw_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
        if not raw_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        api_key = _sanitize_api_key(raw_key)
        _clean_openai_env_nonascii()
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat_json(self, system: str, user: str) -> str:
        from openai import AuthenticationError  # type: ignore
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type":"json_object"},
            )
            return resp.choices[0].message.content.strip()
        except AuthenticationError:
            raise
        except Exception:
            # Фолбэк без строгого JSON — пусть модель вернёт текст, мы распарсим
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user + "\n\nВерни СТРОГО один JSON-объект."}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

# ───────────────────────────────────────────────────────────────────────────
# Картинки: openai | placeholder
class ImageBackend:
    """
    backends:
      - openai   : генерим через gpt-image-1 (нужна верифицированная организация)
      - commons  : ищем подходящее изображение в Wikimedia Commons (без ключей)
      - auto     : пробуем openai → fallback на commons
      - placeholder : как сейчас (1x1 PNG)
    """
    def __init__(self):
        self.backend = (os.getenv("IMAGE_BACKEND") or "placeholder").lower()
        self.model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.size = os.getenv("IMAGE_SIZE", "1024x1024")
        self.embed_data_url = (os.getenv("IMAGE_EMBED_DATA_URL", "true").lower() == "true")

        # куда кладём реальные файлы (если embed_data_url=false)
        self.static_dir = pathlib.Path("static/news_images")
        self.static_dir.mkdir(parents=True, exist_ok=True)

        # клиент OpenAI (если нужен)
        self.client = None
        if self.backend in ("openai", "auto"):
            try:
                raw_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
                if raw_key:
                    api_key = _sanitize_api_key(raw_key)
                    from openai import OpenAI  # type: ignore
                    self.client = OpenAI(api_key=api_key)
                else:
                    if self.backend == "openai":
                        print("[warn] IMAGE_BACKEND=openai, но OPENAI_API_KEY отсутствует → fallback=placeholder")
                        self.backend = "placeholder"
            except Exception as e:
                print("[warn] image backend init failed, fallback to commons:", e)
                self.backend = "commons"

    # ---------- placeholder ----------
    def _placeholder_data_url(self) -> str:
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc``\x00\x00\x00\x04"
            b"\x00\x01\x0b\xe7\x02\x9e\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(tiny_png).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # ---------- commons ----------
    def _search_commons_url(self, query: str) -> Optional[str]:
        """Ищем файл в Wikimedia Commons и возвращаем URL (thumb или оригинал)."""
        try:
            # 1) поиск по файловому пространству имён (namespace=6 — File:)
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 6,
                "srlimit": 10,
                "format": "json",
                "origin": "*",
            }
            r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            hits = data.get("query", {}).get("search", []) or []
            for h in hits:
                title = h.get("title")  # типа "File:Something.png"
                if not title:
                    continue
                # 2) берём инфо по файлу (thumb 1024, либо full url)
                p2 = {
                    "action": "query",
                    "titles": title,
                    "prop": "imageinfo",
                    "iiprop": "url|mime",
                    "iiurlwidth": "1024",
                    "format": "json",
                    "origin": "*",
                }
                r2 = requests.get("https://commons.wikimedia.org/w/api.php", params=p2, timeout=8)
                r2.raise_for_status()
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {}) or {}
                for _, page in pages.items():
                    infos = page.get("imageinfo") or []
                    if not infos:
                        continue
                    info = infos[0]
                    url = info.get("thumburl") or info.get("url")
                    if url and isinstance(url, str) and url.lower().startswith("http"):
                        return url
        except Exception as e:
            print("[warn] commons search failed:", e)
        return None

    def _download_to_static(self, url: str, slug_hint: str) -> Optional[str]:
        """Скачиваем URL в static/news_images/<slug>.<ext> и возвращаем web-путь, либо None."""
        try:
            r = requests.get(url, timeout=12, stream=True)
            r.raise_for_status()
            # определить расширение
            ext = None
            ctype = r.headers.get("Content-Type", "").lower()
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"
            elif "png" in ctype:
                ext = ".png"
            elif "webp" in ctype:
                ext = ".webp"
            else:
                # из URL
                path = urllib.parse.urlparse(url).path
                _, ext = os.path.splitext(path)
                if not ext:
                    ext = ".jpg"
            fname = f"{slug_hint}{ext}"
            fpath = self.static_dir / fname
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return f"/static/news_images/{fname}"
        except Exception as e:
            print("[warn] download failed:", e)
            return None

    # ---------- openai ----------
    def _openai_image(self, topic: str, slug_hint: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            prompt = f"Editorial illustration for a Russian future news article about: {topic}. Minimalist, news style."
            res = self.client.images.generate(model=self.model, prompt=prompt, size=self.size)
            b64 = res.data[0].b64_json
            if self.embed_data_url:
                return f"data:image/png;base64,{b64}"
            else:
                data = base64.b64decode(b64)
                path = self.static_dir / f"{slug_hint}.png"
                with open(path, "wb") as f:
                    f.write(data)
                return f"/static/news_images/{slug_hint}.png"
        except Exception as e:
            print("[warn] openai image failed:", e)
            return None

    # ---------- основной интерфейс ----------
    def generate(self, topic: str, slug_hint: str) -> Tuple[str, bool]:
        """
        Возвращает (html, inline) — <figure><img .../></figure> и флаг inline, если data URL.
        """
        alt = f"иллюстрация: {topic}"
        src: Optional[str] = None

        if self.backend == "openai":
            src = self._openai_image(topic, slug_hint)
        elif self.backend == "commons":
            url = self._search_commons_url(topic)
            if url:
                if self.embed_data_url:
                    # инлайнить как data-url (дороже по размеру ответа; обычно не надо)
                    b = requests.get(url, timeout=12).content
                    b64 = base64.b64encode(b).decode("ascii")
                    src = f"data:image/{('png' if url.endswith('.png') else 'jpeg')};base64,{b64}"
                else:
                    src = self._download_to_static(url, slug_hint)
        elif self.backend == "auto":
            # пробуем openai → commons → placeholder
            src = self._openai_image(topic, slug_hint)
            if not src:
                url = self._search_commons_url(topic)
                if url:
                    if self.embed_data_url:
                        b = requests.get(url, timeout=12).content
                        b64 = base64.b64encode(b).decode("ascii")
                        src = f"data:image/{('png' if url.endswith('.png') else 'jpeg')};base64,{b64}"
                    else:
                        src = self._download_to_static(url, slug_hint)

        if not src:
            # финальный фолбэк — прозрачный пиксель
            if self.embed_data_url:
                src = self._placeholder_data_url()
            else:
                # положим прозрачный PNG в static, чтобы ссылка не была 404
                data_url = self._placeholder_data_url()
                b = base64.b64decode(data_url.split(",",1)[1])
                path = self.static_dir / f"{slug_hint}.png"
                try:
                    with open(path, "wb") as f:
                        f.write(b)
                    src = f"/static/news_images/{slug_hint}.png"
                except Exception:
                    src = data_url

        inline = src.startswith("data:")
        return f'<figure><img src="{src}" alt="{alt}"/></figure>', inline

# Парсинг JSON от модели
def parse_json_or_fallback(raw: str, topic: str) -> Dict[str, Any]:
    try:
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        data = json.loads(s)
    except Exception:
        title = (raw.split("\n", 1)[0] or topic).strip()[:120]
        body = "<p>" + re.sub(r"\n{2,}", "</p><p>", raw).strip() + "</p>"
        data = {"title": title, "section": "list", "tags": topic, "text": body}
    data["title"] = (data.get("title") or topic)[:120]
    sec = (data.get("section") or "list")
    data["section"] = sec if sec in ("main","list","side") else "list"
    text = (data.get("text") or "").strip()
    if not text.startswith("<"):
        text = "<p>" + text.replace("\n\n","</p><p>").replace("\n"," ") + "</p>"
    data["text"] = text
    data["tags"] = (data.get("tags") or topic)
    return data

# ───────────────────────────────────────────────────────────────────────────
# Генерация одной статьи
def generate_one(chat: OpenAIChat, images: ImageBackend, topic: str, context: str) -> Dict[str, Any]:
    user_prompt = USER_TMPL.format(topic=topic, context=context)
    raw = chat.chat_json(SYSTEM_PROMPT, user_prompt)
    data = parse_json_or_fallback(raw, topic)
    # теги — добавим характерные, без дублей
    extra_tags = "Лакан,Жижек,Смулянский,психоанализ,идеология"
    seen = set()
    merged = []
    for t in (str(data.get("tags") or topic) + "," + extra_tags).split(","):
        tt = t.strip()
        if tt and tt.lower() not in seen:
            seen.add(tt.lower()); merged.append(tt)
    data["tags"] = ",".join(merged)

    data["slug"] = slugify(data["title"])
    data["created_at"] = datetime.utcnow().isoformat()

    # картинка (inline data-url или файл в static/)
    img_html, inline = images.generate(topic=data["title"], slug_hint=data["slug"])
    data["text"] = img_html + data["text"]
    data["image_inline"] = inline
    return data

# ───────────────────────────────────────────────────────────────────────────
# Запись payload в файл (для дебага/миграций)
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
# Публичный API для блюпринта /newsgen/run
__all__ = ["run", "OpenAIChat", "ImageBackend", "build_context", "derive_topics", "fetch_recent_articles_from_db"]

def run(
    n: int = 3,
    last_k: int | None = None,
    half_life: int | None = None,
    ctx_max_chars: int | None = None,
    do_import: bool = False,
    topics_override: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Генерит N статей и (опционально) импортирует в БД.
    Возвращает dict: {articles, topics, context, imported}
    """
    # параметры
    last_k        = int(last_k if last_k is not None else getenv_int("LAST_K", 40))
    half_life     = int(half_life if half_life is not None else getenv_int("HALF_LIFE", 10))
    ctx_max_chars = int(ctx_max_chars if ctx_max_chars is not None else getenv_int("CTX_MAX_CHARS", 8000))
    max_tokens    = getenv_int("MAX_TOKENS", 1024)
    try:
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7
    model_id      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # история → контекст и темы
    history = fetch_recent_articles_from_db(limit=last_k)
    context = build_context(history, last_k=last_k, half_life=half_life, max_chars=ctx_max_chars)
    topics  = topics_override or derive_topics(history, n=n, last_k=last_k, half_life=half_life)

    # клиенты
    chat   = OpenAIChat(model=model_id, max_tokens=max_tokens, temperature=temperature)
    images = ImageBackend()

    # генерация
    articles: List[Dict[str, Any]] = []
    for i, t in enumerate(topics):
        art = generate_one(chat, images, t, context)
        art["section"] = "main" if i == 0 else "list"
        if not art.get("created_at"):
            art["created_at"] = datetime.utcnow().isoformat()
        articles.append(art)

    # payload + импорт
    write_payload(articles, path="scripts/articles_payload.py")

    imported = False
    if do_import:
        try:
            from scripts.import_articles import import_articles  # type: ignore
            import_articles(articles)
            imported = True
        except Exception as e:
            print("[warn] import_articles failed:", e)

    return {
        "articles": articles,
        "topics": topics,
        "context": context,
        "imported": imported,
    }

# ───────────────────────────────────────────────────────────────────────────
# CLI (локальный запуск)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--last-k", type=int, dest="last_k")
    p.add_argument("--half-life", type=int, dest="half_life")
    p.add_argument("--ctx-max-chars", type=int, dest="ctx_max_chars")
    p.add_argument("--import", dest="do_import", action="store_true")
    args = p.parse_args()

    out = run(
        n=args.n,
        last_k=args.last_k,
        half_life=args.half_life,
        ctx_max_chars=args.ctx_max_chars,
        do_import=args.do_import,
    )
    print(json.dumps(out, ensure_ascii=False)[:1000])
