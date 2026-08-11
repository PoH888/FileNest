FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r ./requirements.txt

COPY backend/alembic.ini ./backend/alembic.ini
COPY backend/alembic ./backend/alembic
COPY backend/app ./backend/app
COPY core ./core

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["sh", "-c", "python -m alembic -c backend/alembic.ini upgrade head && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
