import pandas as pd
from fastapi import APIRouter, HTTPException
from src.utils.logger import logger
from pydantic import BaseModel
from typing import List
from config import MODEL_CONFIG
from src.utils import model_loader
router = APIRouter()

# 全局变量存储模型和配置
ensemble_model = None
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

@router.get("/health", summary="服务状态查询")
def health_check():
    components = [
        model_loader.ensemble_model,
        model_loader.scaler
    ]
    loaded_status = all(c is not None for c in components)
    """检查API服务状态"""
    return {"status": "alive", "model_loaded": loaded_status}


@router.post("/predict", summary="执行检测")
def predict(data: FeatureSchema):
    """接收特征JSON，直接进行预测"""
    try:
        # 转为 DataFrame
        processed_features = {}

        # 提取基础特征
        base_data = data.model_dump()
        for field in data.model_fields:
            if field not in ["byte_histogram", "top_50_api_2gram"]:
                processed_features[field] = base_data[field]

        # 展开 256 维字节直方图
        for i in range(256):
            processed_features[f"byte_hist_{i}"] = data.byte_histogram[i]

        # 展开 50 维 API 2-gram
        for i in range(50):
            processed_features[f"api_2gram_{i}"] = data.top_50_api_2gram[i]

        # 转换为 DataFrame
        X_df = pd.DataFrame([processed_features])

        X_scaled = model_loader.scaler.transform(X_df)

        # 执行模型推理
        proba = model_loader.ensemble_model.predict_proba(X_scaled)
        score = float(proba[0])

        decision_threshold = MODEL_CONFIG["DECISION_THRESHOLD"]
        is_malicious = score >= decision_threshold

        logger.info(f"预测完成 - 分数: {score}")

        return {
            "success": True,
            "score": round(score, 4),
            "is_malicious": is_malicious,
            "threshold": decision_threshold
        }

    except Exception as e:
        logger.error(f"预测异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")