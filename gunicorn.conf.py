# gunicorn.conf.py
import os

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)

bind     = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers  = env_int("WEB_CONCURRENCY", 2)
timeout  = env_int("GUNICORN_TIMEOUT", 120)
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
