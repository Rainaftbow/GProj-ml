from .base_model import BaseModel
from catboost import CatBoostClassifier

class CatBoostModel(BaseModel):
    """CatBoost分类器"""
    
    def __init__(self):
        super().__init__("catboost")
        self.model = CatBoostClassifier(silent=True)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("CatBoost模型训练完成")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)