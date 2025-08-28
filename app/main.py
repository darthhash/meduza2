from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from . import db
from datetime import datetime
from sqlalchemy.exc import IntegrityError

main_bp = Blueprint("main", __name__)

from .models import Article  # If you move Article to models.py, otherwise import from . if still in __init__.py

@main_bp.get("/")
def index():
    main_obj = Article.query.filter_by(section="main").order_by(Article.created_at.desc()).first()
    side_objs = Article.query.filter_by(section="side").order_by(Article.created_at.desc()).limit(6).all()
    list_objs = Article.query.filter(Article.section != "main").order_by(Article.created_at.desc()).all()
    news = {"main": main_obj, "side": side_objs, "list": list_objs}
    return render_template("index.html", news=news)

@main_bp.get("/news/<slug>")
def article(slug):
    a = Article.query.filter_by(slug=slug).first()
    if not a:
        abort(404)
    return render_template("article.html", article=a)

@main_bp.get("/admin")
def admin():
    items = Article.query.order_by(Article.created_at.desc()).all()
    return render_template("admin.html", items=items)

@main_bp.route("/admin/new", methods=["GET","POST"])
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
            return redirect(url_for("main.admin"))
        except IntegrityError:
            db.session.rollback()
            flash("Такой slug уже есть", "error")
    return render_template("admin_edit.html", article=None)

@main_bp.route("/admin/<int:aid>/edit", methods=["GET","POST"])
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
            return redirect(url_for("main.admin"))
        except IntegrityError:
            db.session.rollback()
            flash("Slug уже используется", "error")
    return render_template("admin_edit.html", article=a)

@main_bp.post("/admin/<int:aid>/delete")
def admin_delete(aid):
    a = Article.query.get_or_404(aid)
    db.session.delete(a)
    db.session.commit()
    flash("Удалено", "success")
    return redirect(url_for("main.admin"))