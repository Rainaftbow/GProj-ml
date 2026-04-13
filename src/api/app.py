from fastapi import FastAPI
from .routes import router as api_router

def create_app():
    """FastAPI应用实例"""
    app = FastAPI(
        title="ML服务",
        description="提供根据特征检测的服务",
    )
    
    app.include_router(api_router, prefix='/api/ml')
    
    return app