# 恶意软件检测系统 - ML模块

## 描述
本模块是恶意软件检测系统的机器学习核心，实现以下功能：
1. 323维特征提取框架
2. 训练五种基础模型（RF、XGBoost、LightGBM、CatBoost、神经网络）
3. 使用软投票集成模型进行预测
4. 基于FastAPI的接口服务

## 目录结构
```
ml/
├── data/                 # 原始数据集
├── logs/                 # 运行日志
├── models_saved/         # 已训练模型
├── src/
│   ├── api/              # API接口
│   ├── feature_extraction/ # 特征提取
│   ├── models/           # 模型实现
│   ├── training/         # 训练实现
│   └── utils/            # 工具函数
├── main.py               # 主入口
├── requirements.txt      # 依赖
└── README.md             # 说明文档
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明

### 训练模型
```bash
python main.py --train
```

### 启动API服务
```bash
python main.py --serve
```

### API接口
- `GET /health`: 健康检查
- `POST /predict`: 接收PE文件进行检测
