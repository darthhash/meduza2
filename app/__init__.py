# app/__init__.py
import os
import importlib.util
import pathlib
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def _load_root_app_module():
    """Загрузить корневой app.py как модуль app_main, если он существует."""
    root = pathlib.Path(__file__).resolve().parent.parent / "app.py"
    if not root.exists():
        return None
    spec = importlib.util.spec_from_file_location("app_main", str(root))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def create_app():
    app = None
    mod = None

    # 1) Пробуем использовать твой корневой app.py (сохранит все старые маршруты главной)
    try:
        mod = _load_root_app_module()
        if mod is not None:
            if hasattr(mod, "create_app") and callable(getattr(mod, "create_app")):
                app = mod.create_app()
            elif hasattr(mod, "app"):
                app = getattr(mod, "app")
    except Exception as e:
        print("[warn] failed to load root app.py:", e)

    # 2) Если корневой app.py не дал app — собираем минимальный
    if app is None:
        app = Flask(__name__, static_folder="static", static_url_path="/static")
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///local.db")
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(app)
        migrate.init_app(app, db)
        # Фолбэк-модель Article, если своей нет
        try:
            from .models import Article  # type: ignore
            globals()["Article"] = Article
        except Exception:
            from sqlalchemy import func
            class Article(db.Model):  # type: ignore
                __tablename__ = "articles"
                id = db.Column(db.Integer, primary_key=True)
                title = db.Column(db.String(255), nullable=False)
                slug = db.Column(db.String(255), nullable=False)
                section = db.Column(db.String(32), default="list")
                tags = db.Column(db.String(1024))
                text = db.Column(db.Text, nullable=False)
                created_at = db.Column(db.DateTime(timezone=True), nullable=False, server_default=func.now())
            globals()["Article"] = Article
    else:
        # 3) Если используем app из корневого app.py — реэкспортируем db/Article при наличии
        if hasattr(mod, "db"):
            globals()["db"] = getattr(mod, "db")
        else:
            app.config.setdefault("SQLALCHEMY_DATABASE_URI", os.getenv("DATABASE_URL", "sqlite:///local.db"))
            app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
            db.init_app(app)
            migrate.init_app(app, db)
        if hasattr(mod, "Article"):
            globals()["Article"] = getattr(mod, "Article")

    # 4) JSON только в UTF-8
    app.config["JSON_AS_ASCII"] = False
    try:
        app.json.ensure_ascii = False
    except Exception:
        pass

    # 5) Регистрируем newsgen, только если его нет
    try:
        if "newsgen" not in app.blueprints:
            from .newsgen import newsgen_bp
            app.register_blueprint(newsgen_bp)
    except Exception as e:
        print("[warn] newsgen not loaded:", e)

    # 6) Health (не мешает твоей главной, другой маршрут)
    if "healthz" not in app.view_functions:
        @app.get("/healthz")
        def healthz():
            return {"ok": True}

    # экспорт для скриптов
    globals()["app"] = app
    return app

# Совместимость с `gunicorn app:app`
try:
    app  # type: ignore
except NameError:
    try:
        app = create_app()  # type: ignore
    except Exception as e:
        print("[warn] auto create_app failed:", e)
