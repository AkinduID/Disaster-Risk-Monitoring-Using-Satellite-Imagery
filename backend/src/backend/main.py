from fastapi import FastAPI

from backend.api import health

app = FastAPI(title="Disaster Risk Monitoring API", version="0.1.0")
app.include_router(health.router)