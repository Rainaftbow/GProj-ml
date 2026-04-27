import joblib
import numpy as np
from src.models.ensemble_model import EnsembleModel
from config import PATH_CONFIG

ensemble_model = None
scaler = None


def load_model():
    global ensemble_model, scaler
    try:
        print("正在初始化集成模型与标准化器...")
        # 实例化并加载模型权重
        ensemble_model = EnsembleModel()
        ensemble_model.load()

        # 加载标准化器
        scaler = joblib.load(PATH_CONFIG["SCALER_PATH"])

        try:
            dummy_input = np.zeros((1, 325))
            # 预热 Scaler
            _ = scaler.transform(dummy_input)
            # 预热模型推理
            _ = ensemble_model.predict_proba(dummy_input)
            print("预热完成")
        except Exception as e:
            print(f"预热失败: {e}")

        print(">>> 推理环境预加载成功")
    except Exception as e:
        print(f">>> 模型加载失败: {str(e)}")
        raise e