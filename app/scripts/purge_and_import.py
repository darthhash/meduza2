# scripts/purge_and_import.py
import os, sys
sys.path.insert(0, os.path.abspath("."))

from app import app, db
from app import Article  # модель берём из app.py (как в моих версиях)
from scripts.import_articles import import_articles

def main():
    with app.app_context():
        # полный сброс таблицы
        db.session.query(Article).delete()
        db.session.commit()
        print("Purged table 'articles'.")

        # перезаливка из payload
        import_articles()
        print("Re-imported from scripts/articles_payload.py")

if __name__ == "__main__":
    main()
