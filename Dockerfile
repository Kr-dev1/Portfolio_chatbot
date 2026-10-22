FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv
RUN UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10 uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "portfolio_chat.main:app", "--host", "0.0.0.0", "--port", "8000"]
