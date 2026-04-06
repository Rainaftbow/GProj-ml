import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.feature_extraction import batch_extractor
from src.models.ensemble_model import EnsembleModel
from src.utils.logger import setup_logging, logger
def load_and_preprocess_data(features_csv, labels_dir):
    """
    加载特征数据和标签数据，并进行预处理
    
    参数:
        features_csv: 特征CSV文件路径
        labels_dir: 标签目录路径
    
    返回:
        X: 特征数据
        y: 标签数据（二分类）
    """
    # 1. 加载特征数据
    print(f"加载特征数据: {features_csv}")
    feature_df = pd.read_csv(features_csv)
    
    # 2. 加载标签数据
    benign_labels = pd.read_csv(os.path.join(labels_dir, "benign.csv"))
    malware_labels = pd.read_csv(os.path.join(labels_dir, "malware.csv"))
    
    # 3. 合并标签数据
    labels_df = pd.concat([benign_labels, malware_labels], ignore_index=True)
    
    # 4. 根据哈希值合并特征和标签
    # 特征数据中的哈希值在file_sha256列
    # 标签数据中的哈希值在hash列
    merged_df = pd.merge(
        feature_df, 
        labels_df[["hash", "malice"]], 
        left_on="file_sha256", 
        right_on="hash",
        how="inner"
    )
    
    # 5. 创建二分类标签（malice阈值0.4）
    merged_df["is_malicious"] = merged_df["malice"].apply(lambda x: 1 if x > 0.4 else 0)
    
    # 6. 分离特征和标签
    # 排除非特征列
    non_feature_cols = ['file_path', 'file_md5', 'file_sha256', 'hash', 'malice', 'is_malicious']
    feature_cols = [col for col in merged_df.columns if col not in non_feature_cols]
    
    X = merged_df[feature_cols].values
    y = merged_df["is_malicious"].values
    
    # 7. 处理缺失值
    # 用该特征列的均值填充缺失值
    for i in range(X.shape[1]):
        col_mean = np.nanmean(X[:, i])
        X[:, i] = np.where(np.isnan(X[:, i]), col_mean, X[:, i])
    
    print(f"数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    return X, y

def preprocess_data(X, y):
    """
    数据预处理：标准化和过采样
    
    参数:
        X: 特征数据
        y: 标签数据
    
    返回:
        X_train, X_test, y_train, y_test: 处理后的数据
    """
    # 1. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. 处理类别不平衡问题（使用SMOTE过采样）
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")
    return X_train, X_test, y_train, y_test

def train_ensemble_model(X_train, y_train, input_dim):
    """训练集成模型"""
    print("\n开始训练集成模型...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_model.train(X_train, y_train)
    print("集成模型训练完成")
    return ensemble_model

def main():
    # 配置日志
    setup_logging()
    logger.info("开始模型训练流程")
    
    # 1. 特征提取（如果尚未完成）
    features_csv = "data/features.csv"
    if not os.path.exists(features_csv):
        print("特征文件不存在，开始提取特征...")
        data_dir = "data/DikeDataset-main/files"
        batch_extractor.extract_dataset_features(data_dir, features_csv)
    
    # 2. 加载数据
    labels_dir = "data/DikeDataset-main/labels"
    X, y = load_and_preprocess_data(features_csv, labels_dir)
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 4. 训练模型
    ensemble_model = train_ensemble_model(X_train, y_train, input_dim=X_train.shape[1])
    
    # 5. 评估模型
    print("\n评估模型性能...")
    ensemble_model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()