from .base_model import BaseModel
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    """XGBoost分类器"""
    
    def __init__(self):
        super().__init__("xgboost")
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("XGBoost模型训练完成")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)