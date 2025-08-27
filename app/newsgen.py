# app/newsgen.py  — замените целиком обработчик /run и /diagnose (остальной файл — как у вас сейчас)
from flask import Blueprint, request, jsonify
import os, hmac, traceback

newsgen_bp = Blueprint("newsgen", __name__, url_prefix="/newsgen")

def _sec_eq(a: str, b: str) -> bool:
    try: return hmac.compare_digest((a or "").strip(), (b or "").strip())
    except Exception: return False

def _check_token(req) -> bool:
    t = (os.getenv("NEWSGEN_TOKEN") or "").strip()
    if not t: return True
    presented = (
        (req.headers.get("X-Token") or "").strip()
        or (req.args.get("token") or "").strip()
        or (req.headers.get("Authorization","").removeprefix("Bearer ").strip())
    )
    return _sec_eq(presented, t)

@newsgen_bp.get("/health")
def health():
    return jsonify(ok=True)

@newsgen_bp.post("/run")
def run_generation():
    if not _check_token(request):
        return jsonify(error="unauthorized"), 401

    payload = request.get_json(silent=True) or {}
    topic = (payload.get("topic") or "").strip()
    if not topic:
        city   = (payload.get("city") or "").strip()
        sector = (payload.get("economy") or payload.get("sector") or "").strip()
        law    = (payload.get("law") or "").strip()
        person = (payload.get("person") or "").strip()
        bits = [b for b in (city, sector or law, person) if b]
        topic = " — ".join(bits) if bits else ""

    n = int(payload.get("n", 1 if topic else 3))
    n = max(1, min(n, 5))

    last_k    = payload.get("last_k")
    half_life = payload.get("half_life")
    ctx_max   = payload.get("ctx_max_chars")
    do_import = bool(payload.get("import", False))

    if "image_size" in payload:
        os.environ["IMAGE_SIZE"] = str(payload["image_size"])
    if "image_embed_data_url" in payload:
        os.environ["IMAGE_EMBED_DATA_URL"] = "true" if payload["image_embed_data_url"] else "false"

    try:
        # preflight ключа: чистим и валидируем
        from scripts.generate_news_openai import _sanitize_api_key, _clean_openai_env_nonascii
        raw_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
        if not raw_key:
            return jsonify(error="missing_openai_key"), 400
        os.environ["OPENAI_API_KEY"] = _sanitize_api_key(raw_key)
        _clean_openai_env_nonascii()

        # дефолт для картинок — placeholder
        os.environ.setdefault("IMAGE_BACKEND", "placeholder")

        # основная генерация
        from scripts.generate_news_openai import run as newsgen_run
        topics_override = [topic] if topic else None
        res = newsgen_run(
            n=n,
            last_k=last_k,
            half_life=half_life,
            ctx_max_chars=ctx_max,
            do_import=do_import,
            topics_override=topics_override
        )
        return jsonify(res), 200

    except BaseException as e:
        # Специально различаем типы
        from openai import AuthenticationError
        if isinstance(e, AuthenticationError):
            return jsonify(error="openai_auth", detail=str(e)), 401
        if isinstance(e, SystemExit):
            # преобразуем в понятную ошибку (до сюда доходить не должно, но на всякий)
            return jsonify(error="system_exit", detail=str(getattr(e, "code", ""))), 500
        # общий случай: вернём и трейс
        return jsonify(error=e.__class__.__name__, detail=str(e), trace=traceback.format_exc(limit=8)), 500

@newsgen_bp.get("/diagnose")
def diagnose():
    if not _check_token(request):
        return jsonify(error="unauthorized"), 401

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    out = {
        "env": {
            "OPENAI_API_KEY_present": bool(api_key),
            "OPENAI_API_KEY_ascii": (api_key.isascii() if api_key else None),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "IMAGE_BACKEND": os.getenv("IMAGE_BACKEND", "placeholder"),
            "IMAGE_SIZE": os.getenv("IMAGE_SIZE", "1024x1024"),
        },
        "db_ok": None,
        "can_generate": None,
    }

    # БД-пинг
    try:
        from app import app as _flask_app, db as _db
        from sqlalchemy import text as sql_text
        with _flask_app.app_context():
            _db.session.execute(sql_text("SET client_encoding TO 'UTF8'"))
            _db.session.execute(sql_text("SELECT 1"))
        out["db_ok"] = True
    except Exception as e:
        out["db_ok"] = f"error: {e}"

    # инициализация клиентов
    try:
        from scripts.generate_news_openai import OpenAIChat, ImageBackend  # type: ignore
        _ = OpenAIChat(os.getenv("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=10)
        _ = ImageBackend()
        out["can_generate"] = True
    except Exception as e:
        out["can_generate"] = f"error: {e}"

    return jsonify(out), 200
