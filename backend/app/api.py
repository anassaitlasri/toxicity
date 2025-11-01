from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .model_registry import registry
from .routers import predict

def create_app() -> FastAPI:
    app = FastAPI(title="Toxicity API", version="0.1.0")

    origins = settings.cors_origins if isinstance(settings.cors_origins, list) else []
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if "*" in origins else origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    @app.on_event("startup")
    async def _load_models():
        registry.load_all()

    app.include_router(predict.router)
    return app
