"""
恶意软件检测系统 - 配置文件
集中管理所有硬编码配置，便于维护和修改
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# ==================== Uvicorn配置 ====================
UVICORN_CONFIG = {
    "App": "main:app",
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "WORKERS": None,
    "PROXY_HEADERS": True,
    "API_PREFIX": "/api/v1",
}

# ==================== 预测配置 ====================
MODEL_CONFIG = {
    # 决策阈值
    "DECISION_THRESHOLD": 0.7, # 业务端最终决策阈值
    "TRAINING_PREDICTION_THRESHOLD": 0.7,  # 训练时的评价阈值
    "TRAINING_DIVIDE_THRESHOLD": 0.4,  # 训练时的恶意标签划分值

    # 集成模型权重 [RF, XGB, LGBM, CatB, NN]
    "ENSEMBLE_WEIGHTS": [0.15, 0.35, 0.35, 0.1, 0.05],

    # 模型参数
    "RANDOM_FOREST_PARAMS": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },

    "XGBOOST_PARAMS": {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },

    "NEURAL_NETWORK_PARAMS": {
        # 网络结构
        "layers": [256, 128, 64, 1],
        "activations": ["relu", "relu", "relu", "sigmoid"],
        "dropout_rates": [0.3, 0.3, 0.0, 0.0],
        
        # 训练参数
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "validation_split": 0.2,
        
        # 优化器和损失函数
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
    },

    "LIGHTGBM_PARAMS": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },

    "CATBOOST_PARAMS": {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False,
        "silent": True,
    },
}

# ==================== 路径配置 ====================
PATH_CONFIG = {
    # 文件后缀过滤
    "SUPPORTED_EXTENSIONS": ('.exe', '.dll', '.sys'),

    # 数据路径
    "DATA_DIR": PROJECT_ROOT / "data",
    "FEATURES_CSV": PROJECT_ROOT / "data" / "features.csv",
    "FILES_DIR": PROJECT_ROOT / "data" / "DikeDataset-main" / "files",
    # 标签路径（load_and_preprocess_data中自定义）
    "LABELS_DIR": PROJECT_ROOT / "data" / "DikeDataset-main" / "labels",

    # 模型保存路径
    "MODELS_SAVED_DIR": PROJECT_ROOT / "models_saved",
    "SCALER_PATH": PROJECT_ROOT / "models_saved" / "scaler.pkl",

    # API组合字典保存路径
    "API_DICT_PATH": PROJECT_ROOT / "models_saved" / "top_50_api_dict.txt",

    # 日志和解释结果路径
    "EXPLANATIONS_DIR": PROJECT_ROOT / "logs" / "explanations",
}

# ==================== 特征提取配置 ====================
FEATURE_CONFIG = {
    # 正常节区名称（PE文件）
    "NORMAL_SECTIONS": {
        b".text", b".data", b".rsrc", b".bss",
        b".rdata", b".reloc", b".idata"
    },

    # 连续可打印字符长度阈值
    "NUM_PRINTABLE_STR_LEN": 4,

    # 可疑字符串模式（正则表达式）
    "SUSPICIOUS_PATTERNS": br"(?i)(cmd\.exe|powershell|http|https|SOFTWARE\\\\|shell|inject)",
}

# ==================== 数据预处理配置 ====================
DATA_CONFIG = {
    # 数据集划分
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,

    # SMOTE过采样
    "SMOTE_RANDOM_STATE": 42,
}

# ==================== SHAP配置 ====================
SHAP_CONFIG = {
    # SHAP解释配置
    "SHAP_BACKGROUND_SIZE": 100,
    "SHAP_SAMPLE_SIZE": 100,

    # 解释结果展示
    # 前20重要性特征显示
    "SHAP_MAX_DISPLAY": 20,
    # 瀑布图样例数量
    "SHAP_WATERFALL_SAMPLES": 5,

    # 特征重要性分析
    "FEATURE_IMPORTANCE_TOP_N": 20,
}