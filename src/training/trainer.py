import os
import pandas as pd
import numpy as np
import argparse
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.feature_extraction import batch_extractor
from src.models.ensemble_model import EnsembleModel
from src.utils.logger import setup_logging, logger
from src.training.shap_explainer import ShapExplainer
from config import MODEL_CONFIG, PATH_CONFIG, DATA_CONFIG, SHAP_CONFIG

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
    merged_df = pd.merge(
        feature_df, 
        labels_df[["hash", "malice"]], 
        left_on="file_sha256", 
        right_on="hash",
        how="inner"
    )
    
    # 5. 创建二分类标签
    merged_df["is_malicious"] = merged_df["malice"].apply(lambda x: 1 if x > MODEL_CONFIG["TRAINING_DIVIDE_THRESHOLD"] else 0)
    
    # 6. 分离特征和标签
    non_feature_cols = ['file_path', 'file_md5', 'file_sha256', 'hash', 'malice', 'is_malicious']
    feature_cols = [col for col in merged_df.columns if col not in non_feature_cols]

    print(feature_cols)

    X = merged_df[feature_cols].values
    y = merged_df["is_malicious"].values
    
    # 7. 处理缺失值
    for i in range(X.shape[1]):
        col_mean = np.nanmean(X[:, i])
        X[:, i] = np.where(np.isnan(X[:, i]), col_mean, X[:, i])
    
    print(f"数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    return X, y, merged_df, feature_cols

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
        X, y, test_size=DATA_CONFIG["TEST_SIZE"], random_state=DATA_CONFIG["RANDOM_STATE"], stratify=y
    )
    
    # 2. 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. 处理类别不平衡问题（使用SMOTE过采样）
    smote = SMOTE(random_state=DATA_CONFIG["SMOTE_RANDOM_STATE"])
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")
    return X_train, X_test, y_train, y_test, scaler

def train_ensemble_model(X_train, y_train, input_dim):
    """训练集成模型"""
    print("\n开始训练集成模型...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_model.train(X_train, y_train)
    print("集成模型训练完成")
    return ensemble_model

def main(do_shap=False):
    # 配置日志
    setup_logging()
    logger.info("开始模型训练流程")
    
    # 创建解释结果保存目录
    explain_dir = PATH_CONFIG["EXPLANATIONS_DIR"]
    os.makedirs(explain_dir, exist_ok=True)
    
    # 1. 特征提取（如果尚未完成）
    features_csv = PATH_CONFIG["FEATURES_CSV"]
    if not os.path.exists(features_csv):
        print("特征文件不存在，开始提取特征...")
        data_dir = PATH_CONFIG["FILES_DIR"]
        batch_extractor.extract_dataset_features(data_dir, features_csv)
    
    # 2. 加载数据
    labels_dir = PATH_CONFIG["LABELS_DIR"]
    X, y, merged_df, feature_cols = load_and_preprocess_data(features_csv, labels_dir)

    # 3. 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    #！！！Scaler持久化
    os.makedirs(PATH_CONFIG["MODELS_SAVED_DIR"], exist_ok=True)
    joblib.dump(scaler, PATH_CONFIG["SCALER_PATH"])
    # 特征名序列查看
    # with open("models_saved/feature_columns.txt", "w") as f:
    #     f.write("\n".join(feature_cols))
    print("Scaler 已保存")
    
    # 4. 训练模型
    ensemble_model = train_ensemble_model(X_train, y_train, input_dim=X_train.shape[1])
    
    # 5. 评估模型
    print("\n评估模型性能...")
    ensemble_model.evaluate(X_test, y_test)
    
    # 6. SHAP模型解释
    if do_shap:
        print("\n开始SHAP模型解释...")
        
        # 获取特征名称
        non_feature_cols = ['file_path', 'file_md5', 'file_sha256', 'hash', 'malice', 'is_malicious']
        feature_names = [col for col in merged_df.columns if col not in non_feature_cols]
        
        # 初始化SHAP解释器（使用训练数据子集作为背景数据）
        background_size = min(SHAP_CONFIG["SHAP_BACKGROUND_SIZE"], len(X_train))
        background_indices = random.sample(range(len(X_train)), background_size)
        X_background = X_train[background_indices]
        
        explainer = ShapExplainer(
            model=ensemble_model,
            X_background=X_background,
            feature_names=feature_names
        )
        
        # 计算SHAP值（使用测试集子集加快计算）
        sample_size = min(SHAP_CONFIG["SHAP_SAMPLE_SIZE"], len(X_test))
        sample_indices = random.sample(range(len(X_test)), sample_size)
        X_sample = X_test[sample_indices]
        
        print(f"计算SHAP值（样本数: {sample_size}）...")
        shap_values = explainer.compute_shap_values(X_sample)
        
        # 生成三种SHAP图表
        print("生成SHAP图表...")
        explainer.generate_bar_plot(
            shap_values, 
            save_path=os.path.join(explain_dir, "shap_bar_plot.png"),
            max_display=SHAP_CONFIG["SHAP_MAX_DISPLAY"]
        )
        
        explainer.generate_beeswarm_plot(
            shap_values, 
            save_path=os.path.join(explain_dir, "shap_beeswarm_plot.png"),
            max_display=SHAP_CONFIG["SHAP_MAX_DISPLAY"]
        )
        
        # 为5个样本生成瀑布图
        for i in range(min(SHAP_CONFIG["SHAP_WATERFALL_SAMPLES"], sample_size)):
            explainer.generate_waterfall_plot(
                shap_values,
                index=i,
                save_path=os.path.join(explain_dir, f"shap_waterfall_sample_{i+1}.png"),
                max_display=SHAP_CONFIG["SHAP_MAX_DISPLAY"]
            )
        
        # 生成特征重要性分析报告
        print("生成特征重要性分析报告...")
        feature_importance = explainer.analyze_feature_importance(shap_values, top_n=SHAP_CONFIG["SHAP_MAX_DISPLAY"])
        report_path = os.path.join(explain_dir, "shap_feature_analysis_report.txt")
        explainer.save_analysis_report(feature_importance, report_path)
        
        print(f"SHAP解释结果已保存至: {explain_dir}")
        print(f"特征重要性分析报告: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恶意软件检测系统 - 训练模块")
    parser.add_argument("--shap", action="store_true", help="执行SHAP模型解释")
    
    args = parser.parse_args()
    main(do_shap=args.shap)
