# 恶意软件检测系统 - ML模块

## 描述
本模块是恶意软件检测系统的机器学习核心 
### 325维特征提取框架
[![peD8RiD.png](https://s41.ax1x.com/2026/04/13/peD8RiD.png)](https://imgchr.com/i/peD8RiD)
### 采用RF、XGBoost、LightGBM、CatBoost、神经网络，通过软投票实现集成模型
[![peD8fRH.png](https://s41.ax1x.com/2026/04/13/peD8fRH.png)](https://imgchr.com/i/peD8fRH)
### 提供基于FastAPI的恶意软件特征检测服务

## 目录结构
```
ml/
├── models_saved/         # 持久化
├── src/
│   ├── api/
│   ├── feature_extraction/ # 特征提取框架实现
│   ├── models/           # 模型实现
│   ├── training/         # 训练实现
│   └── utils/
├── main.py
├── requirements.txt
└── README.md
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明

### 特征提取
```bash
python ./src/feature_extraction/batch_extractor.py
```

### 训练
```bash
python main.py --train
```

### SHAP
```bash
python ./src/training/trainer.py --shap
```


### 启动API服务
```bash
python main.py --serve
```

### API接口
- `GET /health`: 服务状态检查
- `POST /predict`: 特征检测
