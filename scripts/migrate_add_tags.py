# scripts/migrate_add_tags.py
import os, sys
sys.path.insert(0, os.path.abspath("."))

from app import app, db
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

def column_exists(engine, table, column):
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols

def main():
    with app.app_context():
        engine = db.engine
        if column_exists(engine, "articles", "tags"):
            print("tags already exists")
            return

        dialect = engine.dialect.name
        if dialect == "postgresql":
            stmt = text("ALTER TABLE articles ADD COLUMN IF NOT EXISTS tags TEXT")
        else:
            # sqlite / прочие
            stmt = text("ALTER TABLE articles ADD COLUMN tags TEXT")

        try:
            db.session.execute(stmt)
            db.session.commit()
            print("added column 'tags'")
        except SQLAlchemyError as e:
            db.session.rollback()
            print("migration failed:", e)

if __name__ == "__main__":
    main()
