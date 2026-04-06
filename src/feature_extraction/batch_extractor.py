import os
from collections import Counter
import pefile
import pandas as pd
from tqdm import tqdm
from .extractor import FeatureExtractor

def extract_dataset_features(data_dir, output_csv, top_50_api_dict=None):
    """
    批量提取数据集特征
    """
    # 收集所有PE文件路径
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.exe', '.dll', '.sys')):  # PE文件扩展名
                file_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(file_paths)} 个PE文件，开始特征提取...")
    
    # 初始化结果列表
    all_features = []
    
    # 遍历所有文件并提取特征
    for file_path in tqdm(file_paths, desc="提取特征"):
        try:
            extractor = FeatureExtractor(file_path, top_50_api_dict=top_50_api_dict)
            features = extractor.extract_all_features()
            
            # 添加文件路径信息
            features["file_path"] = file_path
            all_features.append(features)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    if not all_features:
        print("未成功提取任何文件特征，请检查输入路径")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_features)
    
    # 处理列表类型的特征
    byte_hist = df.pop("byte_histogram")
    api_2gram = df.pop("top_50_api_2gram")
    
    # 字节直方图特征 (256维)
    for i in range(256):
        df[f"byte_hist_{i}"] = [hist[i] for hist in byte_hist]
    
    # API 2-gram特征 (50维)
    for i in range(50):
        df[f"api_2gram_{i}"] = [gram[i] for gram in api_2gram]
    
    # 保存到CSV
    df.to_csv(output_csv, index=False)
    print(f"特征提取完成，结果保存至: {output_csv}")
    return output_csv

def generate_top50_api_dict(data_dir, output_path):
    """
    从数据集中预先统计出高频率API组合的 top_50_api_2gram 字典
    """
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".exe", ".dll", ".sys")):
                file_paths.append(os.path.join(root, file))

    print(f"\n扫描 {len(file_paths)} 个文件以生成 高频API 2-gram组合字典...")

    all_2grams = []

    # 2-gram提取
    for file_path in tqdm(file_paths, desc="提取 API 链"):
        try:
            with pefile.PE(file_path, fast_load=False) as pe:
                api_list = []
                # 提取导入表
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in entry.imports:
                            if imp.name:
                                api_list.append(
                                    imp.name.decode("utf-8", errors="ignore")
                                )

                # 如果 API 链条长度大于等于2，则生成 2-gram 并拼接成字符串
                if len(api_list) >= 2:
                    file_2grams = [
                        f"{api_list[i]}_{api_list[i+1]}"
                        for i in range(len(api_list) - 1)
                    ]
                    all_2grams.extend(file_2grams)
        except Exception:
            # 忽略损坏PE文件
            continue

    if not all_2grams:
        print("错误：未在任何文件中成功提取到 API组合！")
        return []

    # 统计出现频率最高的前 50 名
    counter = Counter(all_2grams)
    # top_50 拿到的是类似 [('LoadLibraryA_GetProcAddress', 500), ...]
    top_50 = [gram for gram, count in counter.most_common(50)]

    # 4. 保存字典（明文字符串）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(top_50))

    print(f"Top 50 API 字典已成功保存至: {output_path}")
    print("高频前 3 名示例:", top_50[:3])

    return top_50

if __name__ == "__main__":
    # data_dir = "data/DikeDataset-main/files"
    # dict_path = "models_saved/top_50_api_dict.txt"
    #
    # generate_top50_api_dict(data_dir, dict_path)

    data_dir = "data/DikeDataset-main/files"
    output_path = "data/features.csv"
    dict_path = r"E:\AAA\GProj\ml\models_saved\top_50_api_dict.txt"
    
    # 加载字典
    if os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            top_50_api_dict = [line.strip() for line in f.readlines()]
        print(f"已加载长度 {len(top_50_api_dict)} 的高频API组合字典")
    else:
        print(f"警告：在 {dict_path} 处未能找到API字典")
        top_50_api_dict = None
    
    extract_dataset_features(data_dir, output_path, top_50_api_dict=top_50_api_dict)
