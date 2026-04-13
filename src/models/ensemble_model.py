import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .base_model import BaseModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .neural_network import NeuralNetworkModel
from config import MODEL_CONFIG

class EnsembleModel(BaseModel):
    """混合集成学习模型（软投票）"""
    
    def __init__(self, input_dim=None):
        super().__init__("ensemble")
        # 初始化所有基础模型
        self.models = [
            RandomForestModel(),
            XGBoostModel(),
            LightGBMModel(),
            CatBoostModel(),
            NeuralNetworkModel(input_dim) if input_dim else None
        ]
        # 移除未初始化的模型
        self.models = [model for model in self.models if model is not None]
        
    def train(self, X_train, y_train):
        """训练所有基础模型"""
        for model in self.models:
            print(f"\n训练 {model.model_name} 模型...")
            model.train(X_train, y_train)
            model.save()
        print("所有基础模型训练完成")
    
    def predict(self, X):
        """集成预测（多数投票）"""
        proba = self.predict_proba(X)
        return (proba > MODEL_CONFIG["TRAINING_PREDICTION_THRESHOLD"]).astype(int)

    def predict_proba(self, X):
        """
        加权软投票：不同模型赋予不同权重
        """
        proba_sum = np.zeros(X.shape[0])

        # 模型顺序：RF, XGB, LGBM, CatB, NN
        # 给擅长表格特征的 XGB 和 LGBM 更高的权重
        weights = MODEL_CONFIG["ENSEMBLE_WEIGHTS"]

        # 权重归一化
        assert abs(sum(weights) - 1.0) < 1e-5, "权重之和必须为 1"

        for i, model in enumerate(self.models):
            if model.model is None:
                model.load()

            model_proba = model.predict_proba(X)

            if model_proba.shape[1] == 2:
                prob_malicious = model_proba[:, 1]
            else:
                prob_malicious = model_proba.flatten()

            proba_sum += prob_malicious * weights[i]

        return proba_sum
    
    def save(self, filename=None):
        """保存所有基础模型"""
        pass
    
    def load(self, filename=None):
        """加载所有基础模型"""
        for model in self.models:
            model.load()
        print("所有基础模型已加载")
    
    def evaluate(self, X_test, y_test):
        """评估集成模型性能"""
        self.load()
        
        # 获取集成预测
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        print("\n集成模型评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics