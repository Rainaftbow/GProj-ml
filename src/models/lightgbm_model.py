from .base_model import BaseModel
from lightgbm import LGBMClassifier

class LightGBMModel(BaseModel):
    """LightGBM分类器"""
    
    def __init__(self):
        super().__init__("lightgbm")
        self.model = LGBMClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("LightGBM模型训练完成")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)