FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
