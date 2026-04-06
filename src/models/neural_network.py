import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """深度学习神经网络分类器"""
    
    def __init__(self, input_dim):
        super().__init__("neural_network")
        self.input_dim = input_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """构建神经网络模型"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=64, validation_split=0.2):
        """训练模型"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        print("神经网络模型训练完成")
        return history
    
    def predict(self, X):
        """预测类别（0或1）"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """预测恶意概率"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filename=None):
        """保存模型（覆盖基类方法）"""
        if filename is None:
            filename = f"{self.model_name}_model.keras"
        
        model_path = os.path.join(self.model_dir, filename)
        self.model.save(model_path)
        print(f"神经网络模型已保存至: {model_path}")
        return model_path
    
    def load(self, filename=None):
        """加载模型（覆盖基类方法）"""
        if filename is None:
            filename = f"{self.model_name}_model.keras"
        
        model_path = os.path.join(self.model_dir, filename)
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"已从 {model_path} 加载神经网络模型")
        else:
            print(f"模型文件 {model_path} 不存在")
        return model_path