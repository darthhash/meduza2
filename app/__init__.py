# app/__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    from dotenv import load_dotenv; load_dotenv()

    app = Flask(__name__, static_folder="static", static_url_path="/static")

    db_url = os.getenv("DATABASE_URL", "sqlite:///local.db")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # чтобы соединение не залипало на Railway
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 300}

    db.init_app(app)

    # модели должны быть импортированы до create_all
    from .models import Article

    # ВАЖНО: создаём таблицы (или используй миграции, если они настроены)
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            print("[warn] db.create_all skipped:", e)

    # блюпринты
    from .main import main_bp
    app.register_blueprint(main_bp)  # даёт маршрут '/'

    from .newsgen import newsgen_bp
    app.register_blueprint(newsgen_bp, url_prefix="/newsgen")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
