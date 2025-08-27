# wsgi.py
import os
try:
    # фабрика приложения
    from app import create_app
    application = create_app()
except Exception:
    # уже созданный app
    from app import app as application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
