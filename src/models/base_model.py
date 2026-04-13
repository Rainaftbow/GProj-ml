import os
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import PATH_CONFIG

class BaseModel(ABC):
    """模板模式基类"""
    
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.model_dir = PATH_CONFIG["MODELS_SAVED_DIR"]
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def train(self, X_train, y_train):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测结果"""
        if hasattr(self.model, 'feature_names_in_') and isinstance(X, pd.DataFrame):
            return self.model.predict(X)
        return self.model.predict(X)
    
    @abstractmethod
    def predict_proba(self, X):
        """软投票预测概率"""
        if hasattr(self.model, 'feature_names_in_') and isinstance(X, pd.DataFrame):
            return self.model.predict_proba(X)
        return self.model.predict_proba(X)
    
    def save(self, filename=None):
        """保存模型"""
        if filename is None:
            filename = f"{self.model_name}_model.pkl"
        
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, model_path)
        print(f"模型已保存至: {model_path}")
        return model_path
    
    def load(self, filename=None):
        """加载模型"""
        if filename is None:
            filename = f"{self.model_name}_model.pkl"
        
        model_path = os.path.join(self.model_dir, filename)
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"已从 {model_path} 加载模型")
        else:
            print(f"模型文件 {model_path} 不存在")
        return model_path
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            print("模型未训练，请先训练模型")
            return
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else y_pred
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        
        print("\n模型评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics