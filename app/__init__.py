# app/__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    from dotenv import load_dotenv; load_dotenv()
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

    # DB URL: postgres on Railway/Render или локальный sqlite
    db_url = os.getenv("DATABASE_URL", "sqlite:///local.db")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Модели
    from .models import Article

    with app.app_context():
        db.create_all()

    # Блюпринты
    from .main import main_bp
    app.register_blueprint(main_bp)

    from .newsgen import newsgen_bp
    app.register_blueprint(newsgen_bp)  # url_prefix уже задан внутри newsgen.py

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
