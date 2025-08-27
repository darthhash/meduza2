# app/__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    
    from dotenv import load_dotenv; load_dotenv()
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    # from .newsgen import newsgen_bp
    # app.register_blueprint(newsgen_bp, url_prefix="/newsgen")
    db_url = os.getenv("DATABASE_URL", "sqlite:///local.db")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Migrate — опционально, но не гасим все ошибки без логов
    try:
        from flask_migrate import Migrate
        Migrate(app, db)
    except ImportError as e:
        print("[warn] flask_migrate disabled:", e)

    # ⬇️ ВАЖНО: импортируй и зарегистрируй твой блюпринт/роуты
    # если у тебя блюпринт в app/newsgen.py:
    try:
        from .newsgen import newsgen_bp
        app.register_blueprint(newsgen_bp)          # без url_prefix, чтобы '/' был корнем
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[error] cannot register newsgen_bp:", e)

    # если у тебя роуты прямо в app.py и нет блюпринта — можно не регистрировать, см. Вариант B

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
