# scripts/fetch_images_auto.py
import os, sys, re, html, time, hashlib
from typing import Optional, Dict, List
from urllib.parse import quote_plus

import requests

sys.path.insert(0, os.path.abspath("."))
from app import app, db, Article  # noqa

OPENVERSE_API = "https://api.openverse.engineering/v1/images/"
WMC_API = "https://commons.wikimedia.org/w/api.php"
UA = "meduza-good-news/1.3 (+https://example.com)"

FIGURE_RE = re.compile(r'<figure[^>]+class="[^"]*article-hero[^"]*"[^>]*>', re.I)
WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)

RU_STOP = {"и","в","во","на","по","к","с","со","о","об","от","до","за","из","у","как","что","это","не","но","или","а","для","при","бы","же","ли","тоже"}
EN_STOP = {"the","and","or","of","to","in","for","on","a","an","is","are","be","as","with","by"}
def keywords_from_title(title: str, max_words: int = 6) -> str:
    s = (title or "").replace("-", " ")
    s = WORD_RE.sub(" ", s).lower().strip()
    words = [w for w in s.split() if len(w) > 2 and w not in RU_STOP and w not in EN_STOP]
    return " ".join(words[:max_words]) or (title or "")

def tags_to_query(tags: str, max_tags: int = 5) -> Optional[str]:
    if not tags: return None
    parts = [p.strip() for p in tags.split(",") if p.strip()]
    if not parts: return None
    return " ".join(parts[:max_tags])

def google_images_url(q: str) -> str:
    return f"https://www.google.com/search?tbm=isch&q={quote_plus(q)}"

def color_from_slug(slug: str) -> str:
    h = hashlib.md5((slug or "").encode("utf-8")).hexdigest()
    r = 200 + int(h[:2], 16) % 40; g = 200 + int(h[2:4], 16) % 40; b = 200 + int(h[4:6], 16) % 40
    return f"rgb({r},{g},{b})"
def make_svg_placeholder(slug: str, title: str) -> str:
    bg = color_from_slug(slug); t = html.escape((title or "")[:60])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{bg}"/><stop offset="100%" stop-color="#ffffff"/></linearGradient></defs>
  <rect width="100%" height="100%" fill="url(#g)"/>
  <text x="48" y="330" font-size="56" font-family="PT Serif, Georgia, serif" fill="#222">{t}</text>
</svg>'''.replace("#","%23").replace("\n","")
    return f"data:image/svg+xml;utf8,{svg}"
def make_placeholder(slug: str, q: str, title: str):
    return {
        "url": make_svg_placeholder(slug, title),
        "title": title or "image",
        "source": "Google Images",
        "landing": google_images_url(q),
        "license": "—",
    }

def search_openverse(query: str) -> Optional[Dict[str, str]]:
    headers = {"User-Agent": UA, "Accept": "application/json"}
    params = {"q": query, "license": "cc0,pdm", "page_size": 3, "mature": "false"}
    r = requests.get(OPENVERSE_API, params=params, headers=headers, timeout=15)
    if r.status_code == 400:
        r = requests.get(OPENVERSE_API, params={**params, "q": keywords_from_title(query)}, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    for it in data.get("results", []):
        url = it.get("url") or it.get("thumbnail")
        if not url: continue
        return {
            "url": url,
            "title": it.get("title") or query,
            "source": it.get("source") or "Openverse",
            "landing": it.get("foreign_landing_url") or url,
            "license": (it.get("license") or "cc0").upper(),
        }
    return None

def search_wikimedia_pd(query: str) -> Optional[Dict[str, str]]:
    headers = {"User-Agent": UA}
    params = {
        "action": "query", "generator": "search", "gsrnamespace": "6", "gsrlimit": "5", "gsrsearch": query,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1200", "format": "json",
        "formatversion": "2", "origin": "*",
    }
    r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=headers, timeout=15)
    if r.status_code == 400:
        params["gsrsearch"] = keywords_from_title(query)
        r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json(); pages = (data.get("query") or {}).get("pages") or []
    for p in pages:
        ii = (p.get("imageinfo") or [])
        if not ii: continue
        info = ii[0]; meta = (info.get("extmetadata") or {})
        lic_short = (meta.get("LicenseShortName") or {}).get("value", "")
        lic = (meta.get("License") or {}).get("value", "")
        ok = ("Public domain" in lic_short) or (lic.upper() == "CC0")
        if not ok: continue
        url = info.get("thumburl") or info.get("url")
        if not url: continue
        title = p.get("title", "Wikimedia image").replace("File:", "")
        landing = "https://commons.wikimedia.org/wiki/" + p.get("title", "")
        return {"url": url, "title": title, "source": "Wikimedia Commons", "landing": landing, "license": lic_short or lic or "Public Domain"}
    return None

def inject_figure(html_text: str, img: Dict[str, str]) -> str:
    if not img or not img.get("url"):
        return html_text or ""
    # уже есть hero — не дублируем
    if FIGURE_RE.search(html_text or ""):
        return html_text or ""

    # подпись оставляем (на главную она больше не попадёт, см. FIGURE_BLOCK_RE)
    cap = (
        'Источник: <a href="{landing}" rel="noopener" target="_blank">{src}</a> {lic}'
        .format(
            landing=html.escape(img["landing"]),
            src=html.escape(img["source"]),
            lic=f"({html.escape(img['license'])})" if img.get("license") else ""
        )
    )
    figure = (
        '<figure class="article-hero">'
        f'<img src="{html.escape(img["url"])}" alt="{html.escape(img["title"])}" '
        'style="display:block;width:100%;height:auto;border-radius:8px" />'
        f"<figcaption>{cap}</figcaption>"
        "</figure>\n"
    )
    return (figure + (html_text or "")).strip()

def main():
    with app.app_context():
        updated = 0
        arts = Article.query.order_by(Article.created_at.desc()).all()
        for a in arts:
            if FIGURE_RE.search(a.text or ""): continue

            # строим поисковый запрос: приоритет — теги
            q = tags_to_query(a.tags) or keywords_from_title(a.title or a.slug.replace("-", " "))

            img = None
            try: img = search_openverse(q)
            except Exception as e: print("openverse error:", e)
            if not img:
                try: img = search_wikimedia_pd(q)
                except Exception as e: print("wikimedia error:", e)
            if not img:
                img = make_placeholder(a.slug, q, a.title or a.slug.replace("-", " "))

            a.text = inject_figure(a.text or "", img)
            db.session.add(a); updated += 1
            time.sleep(0.3)

        db.session.commit()
        print(f"done, updated {updated} articles")
    

if __name__ == "__main__":
    main()
