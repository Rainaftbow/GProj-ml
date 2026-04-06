from fastapi import FastAPI
from .routes import router as api_router

def create_app():
    """FastAPI应用实例"""
    app = FastAPI(
        title="恶意软件检测API",
        description="基于机器学习的恶意软件检测系统API",
        version="1.0.0"
    )
    
    # 包含路由
    app.include_router(api_router)
    
    return app