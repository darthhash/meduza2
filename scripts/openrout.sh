
export OPENAI_URL="https://openrouter.ai/api/v1/chat/completions"

export OPENAI_MODEL="qwen/qwen3-coder:free"

export OPENAI_API_KEY="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"

export DISABLE_IMAGES=1

export OPENROUTER_SITE="https://your-site.example"   # любой твой домен/URL проекта
export OPENROUTER_TITLE="meduza-good-news"

N=1 python scripts/generate_news.py
