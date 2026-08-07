from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import create_directories
from app.routes import router


def create_app() -> FastAPI:
    create_directories()

    app = FastAPI(
        title="LipVision",
        description="AI-Powered Lip Reading",
        version="2.0.0",
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    app.include_router(router)

    return app


app = create_app()