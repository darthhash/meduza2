# 0) ОБНОВИ .gitignore, чтобы мусор не попадал в образ/репо
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
data/models/
*.gguf
static/news_images/
.env
EOF

# 1) requirements.txt (ГАРАНТИЯ, что gunicorn будет установлен)
cat > requirements.txt <<'EOF'
Flask>=3.0
SQLAlchemy>=2.0
psycopg2-binary>=2.9
openai>=1.30.0
python-dotenv>=1.0
gunicorn>=21.2
# если у тебя есть другие зависимости из проекта — ДОБАВЬ их сюда
EOF

# 2) wsgi.py — единая точка входа для gunicorn
cat > wsgi.py <<'PY'
import os
try:
    # если у тебя фабрика приложений
    from app import create_app
    application = create_app()
except Exception:
    # иначе берём уже созданный app
    from app import app as application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
PY

# 3) Dockerfile — принуждаем Railway собирать Python-образ с gunicorn
cat > Dockerfile <<'DOCKER'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway прокидывает PORT; по умолчанию 8000
ENV PORT=8000
CMD ["python","-m","gunicorn","wsgi:application","--bind","0.0.0.0:${PORT}","--workers","2","--threads","4","--timeout","120"]
DOCKER

# 4) .dockerignore — чтобы не тащить гигабайты в билд-контекст
cat > .dockerignore <<'EOF'
.git
.venv
__pycache__
*.pyc
data/models
*.gguf
static/news_images
.env
EOF

# 5) (опционально) Procfile — если вдруг уберёшь Dockerfile позже
cat > Procfile <<'EOF'
web: python -m gunicorn "wsgi:application" --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120
EOF

git add -A
git commit -m "fix: deterministic Railway deploy (Dockerfile + gunicorn + wsgi)"
git push -u origin $(git branch --show-current)
