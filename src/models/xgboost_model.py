from .base_model import BaseModel
from xgboost import XGBClassifier
from config import MODEL_CONFIG

class XGBoostModel(BaseModel):
    """XGBoost分类器"""
    
    def __init__(self):
        super().__init__("xgboost")
        # 从配置文件中获取参数
        xgb_params = MODEL_CONFIG["XGBOOST_PARAMS"]
        self.model = XGBClassifier(**xgb_params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("XGBoost模型训练完成")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)