FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 -r requirements.txt

COPY app ./app

# Run as a non-root user; /models is where the embedding models are cached (a volume).
RUN useradd --create-home appuser && mkdir /models && chown appuser /models
ENV FASTEMBED_CACHE_PATH=/models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
