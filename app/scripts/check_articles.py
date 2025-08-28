import os, sys
sys.path.insert(0, os.path.abspath("."))

from app import create_app, db
from app.models import Article

def main():
    app = create_app()
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