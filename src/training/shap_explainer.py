"""
SHAP模型可解释性模块
提供对机器学习模型预测结果的可解释性分析
生成三种核心图表：条形图、蜂群图和瀑布图
"""
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

class ShapExplainer:
    def __init__(self, model, X_background, feature_names):
        """
        初始化SHAP解释器
        :param model: 训练好的机器学习模型
        :param X_background: 背景数据集（用于SHAP解释器）
        :param feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

        self.explainer = shap.Explainer(
            model.predict_proba,
            X_background,
            feature_names=feature_names
        )

    def compute_shap_values(self, X):
        """
        计算SHAP值
        :param X: 输入数据 (n_samples, n_features)
        :return: SHAP值对象
        """
        num_features = X.shape[1]
        max_evals = 2 * num_features + 10
        return self.explainer(X, max_evals=max_evals)

    def generate_bar_plot(self, shap_values, save_path=None, max_display=20):
        """
        生成特征重要性条形图
        """
        # 前20名特征的索引
        inds = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1]

        # 切出前20名，去除其他特征总和
        shap_values_sliced = shap_values[:, inds[:max_display]]

        # 正常的画图和保存逻辑
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values_sliced, max_display=max_display, show=False)

        plt.title(f"SHAP特征重要性 (Top {max_display})", fontsize=14, pad=15)
        plt.tight_layout()

        self._save_or_show(save_path)

    def generate_beeswarm_plot(self, shap_values, save_path=None, max_display=20):
        """
        生成蜂群图（宏观特征机理分析）
        """
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.title("SHAP蜂群图 (Top 20)", fontsize=14, pad=15)
        plt.tight_layout()

        self._save_or_show(save_path)

    def generate_waterfall_plot(self, shap_values, index, save_path=None, max_display=20):
        """
        生成瀑布图（单样本局部溯源拆解）
        """
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
        plt.title(f"SHAP瀑布图 (样本 {index})", fontsize=14, pad=15)
        plt.tight_layout()

        self._save_or_show(save_path)

    def _save_or_show(self, save_path):
        """
        私有辅助方法：保存图片或展示图片
        """
        if save_path:
            # 自动创建不存在的文件夹
            dirname = os.path.dirname(save_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches防止边缘文字被切掉
            plt.close()
        else:
            plt.show()

    def analyze_feature_importance(self, shap_values, top_n=20):
        """
        分析特征重要性并返回结果
        """
        # 计算平均绝对SHAP值
        abs_shap = np.abs(shap_values.values).mean(axis=0)

        # 获取特征重要性排序
        sorted_indices = np.argsort(abs_shap)[::-1]
        top_indices = sorted_indices[:top_n]

        # 创建结果字典
        feature_importance = {}
        for i in top_indices:
            feature_name = self.feature_names[i]
            feature_importance[feature_name] = {
                "mean_abs_shap": abs_shap[i],
                "feature_type": self._classify_feature(feature_name)
            }

        return feature_importance

    def _classify_feature(self, feature_name):
        """
        分类特征类型 (PE特征或API特征)
        """
        # 兼容你的 2-gram 命名
        if "api_" in feature_name.lower() or "gram" in feature_name.lower():
            return "API特征"
        return "PE特征"

    def save_analysis_report(self, feature_importance, report_path):
        """
        保存特征重要性分析报告（毕业论文里凑字数和贴表格的绝佳材料！）
        """
        dirname = os.path.dirname(report_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SHAP特征重要性分析报告\n")
            f.write("=" * 40 + "\n\n")

            # 分析PE特征
            f.write("PE特征分析:\n")
            f.write("-" * 40 + "\n")
            pe_features = {k: v for k, v in feature_importance.items() if v['feature_type'] == 'PE特征'}
            for i, (feature, info) in enumerate(pe_features.items(), 1):
                f.write(f"{i}. {feature}: 平均绝对SHAP值 = {info['mean_abs_shap']:.4f}\n")

            # 分析API特征
            f.write("\nAPI特征分析:\n")
            f.write("-" * 40 + "\n")
            api_features = {k: v for k, v in feature_importance.items() if v['feature_type'] == 'API特征'}
            for i, (feature, info) in enumerate(api_features.items(), 1):
                f.write(f"{i}. {feature}: 平均绝对SHAP值 = {info['mean_abs_shap']:.4f}\n")

            # 总体总结
            f.write("\n总结:\n")
            f.write("-" * 40 + "\n")
            total_pe_importance = sum(info['mean_abs_shap'] for info in pe_features.values())
            total_api_importance = sum(info['mean_abs_shap'] for info in api_features.values())
            f.write(f"PE特征总贡献度: {total_pe_importance:.4f}\n")
            f.write(f"API特征总贡献度: {total_api_importance:.4f}\n")
            # 加个保护，防止分母为 0
            ratio = total_pe_importance / total_api_importance if total_api_importance != 0 else float('inf')
            f.write(f"PE/API特征贡献比: {ratio:.2f}\n")