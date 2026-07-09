from fastapi import FastAPI

app = FastAPI(title="FileNest API")

@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}