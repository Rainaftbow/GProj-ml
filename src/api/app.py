from fastapi import FastAPI
from .routes import router as api_router
from config import UVICORN_CONFIG
from contextlib import asynccontextmanager
from src.utils.model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

def create_app():
    """FastAPI应用实例"""
    app = FastAPI(
        title="ML服务",
        description="提供根据特征检测的服务",
        lifespan=lifespan
    )
    
    app.include_router(api_router, prefix=UVICORN_CONFIG["API_PREFIX"])
    
    return app