# app/newsgen.py
from flask import Blueprint, request, jsonify
import os, hmac, traceback

newsgen_bp = Blueprint("newsgen", __name__, url_prefix="/newsgen")

def _sec_eq(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest((a or "").strip(), (b or "").strip())
    except Exception:
        return False

def _check_token(req) -> bool:
    t = (os.getenv("NEWSGEN_TOKEN") or "").strip()
    if not t:
        return True
    presented = (
        (req.headers.get("X-Token") or "").strip()
        or (req.args.get("token") or "").strip()
        or (req.json or {}).get("token", "")
    )
    return _sec_eq(t, presented)

@newsgen_bp.get("/diagnose")
def diagnose():
    if not _check_token(request):
        return jsonify(error="unauthorized"), 401

    out = {
        "env": {
            "OPENAI_KEY_present": bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "DATABASE_URL_present": bool(os.getenv("DATABASE_URL")),
        },
        "db_ok": False,
        "can_generate": False,
    }

    # Проверка БД
    try:
        from . import db as _db
        from sqlalchemy import text as sql_text
        _db.session.execute(sql_text("SELECT 1"))
        out["db_ok"] = True
    except Exception as e:
        out["db_ok"] = f"error: {e}"

    # Проверка клиента OpenAI
    try:
        from openai import OpenAI  # noqa
        _ = OpenAI()
        out["can_generate"] = True
    except Exception as e:
        out["can_generate"] = f"error: {e}"

    return jsonify(out), 200

# app/newsgen.py (фрагмент внутри run_newsgen)
@newsgen_bp.post("/run")
def run_newsgen():
    if not _check_token(request):
        return jsonify(error="unauthorized"), 401

    body = request.get_json(silent=True) or {}
    n = int(body.get("n") or 3)
    do_import = bool(body.get("import") or body.get("save") or False)

    # Новые опции:
    lam = float(body.get("lambda") or body.get("lam") or 0.10)          # 0.06..0.15
    ctx_items = int(body.get("context_items") or 60)                     # 30..100
    model = (body.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

    try:
        from .scripts.generate_news_openai import run as gen_run
        articles = gen_run(n=n, model=model, lam=lam, context_items=ctx_items)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=f"generation failed: {e}"), 500

    inserted_ids = []
    if do_import:
        try:
            from .scripts.import_articles import import_articles_from_list
            inserted_ids = import_articles_from_list(articles)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(error=f"db import failed: {e}", draft=articles), 500

    return jsonify({
        "ok": True,
        "generated": len(articles),
        "imported": len(inserted_ids),
        "ids": inserted_ids,
        "preview": [{"id": i, **a} for i, a in zip(inserted_ids, articles)]
    }), 200
