import os
import json
import pandas as pd
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from src.models.ensemble_model import EnsembleModel
from src.feature_extraction.extractor import FeatureExtractor
from src.utils.logger import logger, setup_logging
router = APIRouter()

# 全局变量存储模型和配置
ensemble_model = None
model_loaded = False
top_50_api_dict = []

def load_model():
    """加载集成模型和API字典"""
    global ensemble_model, model_loaded, top_50_api_dict
    
    # 确保日志系统已配置
    if not logger.handlers:
        setup_logging()
    
    if not model_loaded:
        try:
            # 加载API字典
            dict_path = "models_saved/top_50_api_dict.txt"
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    top_50_api_dict = [line.strip() for line in f.readlines()]
            
            # 加载集成模型
            ensemble_model = EnsembleModel()
            ensemble_model.load()
            
            model_loaded = True
            logger.info("模型和API字典加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise HTTPException(status_code=500, detail="模型加载失败")

@router.get("/health", summary="服务健康检查")
async def health_check():
    """检查API服务状态"""
    return {"status": "alive", "model_loaded": model_loaded}


@router.post("/predict", summary="执行恶意软件检测")
async def predict(file: UploadFile = File(...)):
    """接收PE文件，提取特征并预测恶意概率"""
    try:
        if not model_loaded:
            load_model()

        # 上传文件持久化
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 特征提取
        try:
            extractor = FeatureExtractor(file_path, top_50_api_dict=top_50_api_dict)
            features = extractor.extract_all_features()
        finally:
            # 临时文件删除
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"无法删除临时文件: {str(e)}")

        processed_features = {}

        # 17维基础特征
        base_features = [
            "file_size",
            "global_entropy",
            "e_magic",
            "machine",
            "number_of_sections",
            "time_date_stamp",
            "address_of_entry_point",
            "image_base",
            "section_alignment",
            "subsystem",
            "max_section_entropy",
            "is_abnormal_section_name",
            "num_imported_dlls",
            "is_export_present",
            "resource_size",
            "num_printable_strings",
            "suspicious_str_count",
        ]
        for feat in base_features:
            processed_features[feat] = features[feat]

        # 256维字节直方图
        for i in range(256):
            processed_features[f"byte_hist_{i}"] = features[
                "byte_histogram"
            ][i]

        # 50维API 2-gram
        for i in range(50):
            processed_features[f"api_2gram_{i}"] = features[
                "top_50_api_2gram"
            ][i]

        X_df = pd.DataFrame([processed_features])

        # 预测恶意概率
        proba = ensemble_model.predict_proba(X_df)
        score = float(proba[0])
        is_malicious = True if score >= 0.36 else False

        return {
            "file_name": file.filename,
            "score": score,
            "is_malicious": is_malicious
        }
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")