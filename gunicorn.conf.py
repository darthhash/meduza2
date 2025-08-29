# gunicorn.conf.py
import os
bind     = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers  = int(os.environ.get("WEB_CONCURRENCY", "2"))
threads  = int(os.environ.get("GUNICORN_THREADS", "4"))
timeout  = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
