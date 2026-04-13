from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_CONFIG

class RandomForestModel(BaseModel):
    """随机森林分类器"""
    
    def __init__(self):
        super().__init__("random_forest")
        # 从配置文件中获取参数
        rf_params = MODEL_CONFIG["RANDOM_FOREST_PARAMS"]
        self.model = RandomForestClassifier(**rf_params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("随机森林模型训练完成")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)