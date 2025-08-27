# scripts/check_articles.py
import os, sys
sys.path.insert(0, os.path.abspath("."))

from app import app, db, Article

def main():
    with app.app_context():
        total = db.session.query(Article).count()
        empty = db.session.query(Article)\
            .filter((Article.text == None) | (Article.text == "")).all()
        print(f"Total: {total}")
        print(f"Empty text: {len(empty)}")
        for a in empty:
            print("-", a.slug, "|", (a.title or "")[:80])

if __name__ == "__main__":
    main()
