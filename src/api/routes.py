import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.models.ensemble_model import EnsembleModel
from src.utils.logger import logger
from pydantic import BaseModel
from typing import List
router = APIRouter()

# 全局变量存储模型和配置
ensemble_model = None
model_loaded = False
top_50_api_dict = []

# 定义Schema
class FeatureSchema(BaseModel):
    file_size: float
    global_entropy: float
    e_magic: int
    machine: int
    number_of_sections: int
    time_date_stamp: int
    address_of_entry_point: int
    image_base: int
    section_alignment: int
    subsystem: int
    is_abnormal_section_name: int
    all_sections_size_ratio: float
    wx_section_ratio: float
    max_section_entropy: float
    num_imported_dlls: int
    is_export_present: int
    resource_size: int
    num_printable_strings: int
    suspicious_str_count: int
    byte_histogram: List[float]  # 256维
    top_50_api_2gram: List[float] # 50维


def load_model():
    global ensemble_model, scaler, feature_columns, model_loaded
    if not model_loaded:
        try:
            # 加载集成模型
            ensemble_model = EnsembleModel()
            ensemble_model.load()

            # 加载标准化器 (Scaler)
            scaler = joblib.load("models_saved/scaler.pkl")

            model_loaded = True
            print("推理环境加载成功：模型、Scaler已就绪")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"加载失败: {str(e)}")

@router.get("/health", summary="服务健康检查")
async def health_check():
    """检查API服务状态"""
    return {"status": "alive", "model_loaded": model_loaded}


@router.post("/predict", summary="执行检测")
async def predict(data: FeatureSchema):
    """接收特征JSON，直接进行预测"""
    try:
        if not model_loaded:
            load_model()

        # 转为 DataFrame
        processed_features = {}

        # 提取基础特征
        base_data = data.model_dump()
        for field in FeatureSchema.model_fields:
            if field not in ["byte_histogram", "top_50_api_2gram"]:
                processed_features[field] = base_data[field]

        # 展开 256 维字节直方图
        for i in range(256):
            processed_features[f"byte_hist_{i}"] = data.byte_histogram[i]

        # 展开 50 维 API 2-gram
        for i in range(50):
            processed_features[f"api_2gram_{i}"] = data.top_50_api_2gram[i]

        # 2. 转换为 DataFrame
        X_df = pd.DataFrame([processed_features])

        X_scaled = scaler.transform(X_df)

        # 3. 执行模型推理
        proba = ensemble_model.predict_proba(X_scaled)
        score = float(proba[0])

        decision_threshold = 0.7
        is_malicious = score >= decision_threshold

        logger.info(f"预测完成 - 分数: {score}")

        return {
            "success": True,
            "score": round(score, 4),
            "is_malicious": is_malicious,
            "threshold": decision_threshold
        }

    except Exception as e:
        logger.error(f"JSON预测异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")