# --- app.py (минимальный каркас, чтобы был глобальный app) ---

import os
from datetime import datetime
from dotenv import load_dotenv; load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ВАЖНО: глобальный объект приложения
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

# БД: SQLite локально, Postgres на проде
db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url or ("sqlite:///" + os.path.join(DATA_DIR, "news.db"))
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Модель
class Article(db.Model):
    __tablename__ = "articles"
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(255), unique=True, index=True, nullable=False)
    title = db.Column(db.String(500), nullable=False)
    text = db.Column(db.Text, default="")
    section = db.Column(db.String(20), default="list")
    tags = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# Разовая инициализация таблиц (без падения, если уже есть)
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print("[warn] db.create_all skipped:", e)

# Подключаем блюпринт newsgen один раз
try:
    from app.newsgen import newsgen_bp
    if 'newsgen' not in app.blueprints:
        app.register_blueprint(newsgen_bp)   # в самом bp задан url_prefix="/newsgen"
except Exception as e:
    import traceback; traceback.print_exc()
    print("[warn] newsgen blueprint not mounted in app.py:", e)

# --- РОУТЫ ДОЛЖНЫ БЫТЬ НА УРОВНЕ МОДУЛЯ (не в __main__) ---

@app.get("/")
def index():
    main_obj = Article.query.filter_by(section="main").order_by(Article.created_at.desc()).first()
    side_objs = Article.query.filter_by(section="side").order_by(Article.created_at.desc()).limit(6).all()
    list_objs = Article.query.filter(Article.section != "main").order_by(Article.created_at.desc()).all()
    news = {"main": main_obj, "side": side_objs, "list": list_objs}
    return render_template("index.html", news=news)

@app.get("/news/<slug>")
def article(slug):
    a = Article.query.filter_by(slug=slug).first()
    if not a:
        abort(404)
    return render_template("article.html", article=a)

@app.get("/admin")
def admin():
    items = Article.query.order_by(Article.created_at.desc()).all()
    return render_template("admin.html", items=items)

@app.route("/admin/new", methods=["GET","POST"])
def admin_new():
    if request.method == "POST":
        from re import sub
        title = (request.form.get("title") or "").strip()
        section = (request.form.get("section") or "list").strip()
        slug = (request.form.get("slug") or "").strip()
        if not slug:
            s = title.lower()
            s = sub(r"[^a-z0-9]+", "-", s)
            slug = sub(r"-{2,}", "-", s).strip("-") or "article"
        text = request.form.get("text") or ""
        tags = (request.form.get("tags") or "").strip()
        a = Article(slug=slug, title=title, section=section, text=text, tags=tags)
        db.session.add(a)
        try:
            db.session.commit()
            flash("Создано", "success")
            return redirect(url_for("admin"))
        except IntegrityError:
            db.session.rollback()
            flash("Такой slug уже есть", "error")
    return render_template("admin_edit.html", article=None)

@app.route("/admin/<int:aid>/edit", methods=["GET","POST"])
def admin_edit(aid):
    a = Article.query.get_or_404(aid)
    if request.method == "POST":
        from re import sub
        a.title = (request.form.get("title") or a.title).strip()
        a.section = (request.form.get("section") or a.section).strip()
        slug = (request.form.get("slug") or a.slug).strip()
        s = slug.lower()
        s = sub(r"[^a-z0-9]+", "-", s)
        a.slug = sub(r"-{2,}", "-", s).strip("-") or a.slug
        a.text = request.form.get("text") or ""
        a.tags = (request.form.get("tags") or "").strip()
        try:
            db.session.commit()
            flash("Сохранено", "success")
            return redirect(url_for("admin"))
        except IntegrityError:
            db.session.rollback()
            flash("Slug уже используется", "error")
    return render_template("admin_edit.html", article=a)

@app.post("/admin/<int:aid>/delete")
def admin_delete(aid):
    a = Article.query.get_or_404(aid)
    db.session.delete(a)
    db.session.commit()
    flash("Удалено", "success")
    return redirect(url_for("admin"))

# Локальный запуск (не мешает gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5010)), debug=True)
