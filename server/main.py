# server/main.py
import os, sys, json, re
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

from scripts.generate_news import (
    LlamaCppBackend, TransformersBackend, read_topics,
    SYSTEM_PROMPT_RU, USER_PROMPT_TMPL, generate_one
)
from app import slugify  # твой транслит
# import_articles можно вызывать по флагу, чтобы сразу писать в БД
from scripts.import_articles import import_articles, import_articles_from_payload_path

app = FastAPI(title="News Generator", version="1.0")

# ---------- инициализация модели один раз ----------
LLM_BACKEND = os.getenv("LLM_BACKEND", "llama").strip().lower()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

llm = None
if LLM_BACKEND == "llama":
    repo_id = os.getenv("GGUF_REPO_ID", "Qwen/Qwen2.5-7B-Instruct-GGUF")
    filename = os.getenv("GGUF_FILENAME", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    llm = LlamaCppBackend(repo_id, filename, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
else:
    model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    llm = TransformersBackend(model_id, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

class GenerateRequest(BaseModel):
    topics: Optional[List[str]] = None
    topics_file: Optional[str] = None
    n: int = 3
    do_import: bool = False

@app.get("/health")
def health():
    return {"ok": True, "model_backend": LLM_BACKEND, "time": datetime.utcnow().isoformat()}

@app.post("/generate")
def generate(req: GenerateRequest):
    topics = req.topics or read_topics(
        cli_topics=",".join(req.topics) if req.topics else None,
        topics_file=req.topics_file
    )
    if req.n and req.n > 0:
        topics = topics[:req.n]

    articles = []
    for t in topics:
        art = generate_one(llm, t)
        articles.append(art)

    # пишем сразу в payload файл, как у тебя в потоке
    from scripts.generate_news import write_payload
    write_payload(articles, path="scripts/articles_payload.py")

    if req.do_import:
        # по твоему же импортеру
        import_articles(articles)

    return {"count": len(articles), "articles": articles}
