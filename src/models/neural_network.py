import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .base_model import BaseModel
from config import MODEL_CONFIG

class NeuralNetworkModel(BaseModel):
    """深度学习神经网络分类器"""
    
    def __init__(self, input_dim):
        super().__init__("neural_network")
        self.input_dim = input_dim
        self.nn_params = MODEL_CONFIG["NEURAL_NETWORK_PARAMS"]
        self.model = self._build_model()
    
    def _build_model(self):
        """构建神经网络模型"""
        layers = []
        
        # 构建网络层
        for i, (units, activation, dropout_rate) in enumerate(
            zip(self.nn_params["layers"], 
                self.nn_params["activations"], 
                self.nn_params["dropout_rates"])
        ):
            if i == 0:
                # 第一层需要指定输入维度
                layers.append(Dense(units, activation=activation, input_shape=(self.input_dim,)))
            else:
                layers.append(Dense(units, activation=activation))
            
            # 添加Dropout层（如果dropout_rate > 0）
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        
        model = Sequential(layers)
        
        # 配置优化器
        if self.nn_params["optimizer"] == "adam":
            optimizer = Adam(learning_rate=self.nn_params["learning_rate"])
        else:
            optimizer = self.nn_params["optimizer"]
        
        model.compile(
            optimizer=optimizer,
            loss=self.nn_params["loss"],
            metrics=self.nn_params["metrics"]
        )
        return model
    
    def train(self, X_train, y_train, epochs=None, batch_size=None, validation_split=None):
        """训练模型"""
        # 如果参数未提供使用配置中的默认值
        if epochs is None:
            epochs = self.nn_params["epochs"]
        if batch_size is None:
            batch_size = self.nn_params["batch_size"]
        if validation_split is None:
            validation_split = self.nn_params["validation_split"]
        
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