import argparse
import uvicorn
import warnings
from src.api.app import create_app
from src.training.trainer import main as train_main

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# FastAPI应用
app = create_app()

def run_train():
    """模型训练"""
    print("启动模型训练...")
    try:
        train_main()
        print("模型训练完成")
    except Exception as e:
        print(f"模型训练失败: {str(e)}")

def run_api():
    """API服务"""
    print("启动API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恶意软件检测系统 - ML模块")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--serve", action="store_true", help="启动API服务")
    
    args = parser.parse_args()
    
    if args.train:
        run_train()
    elif args.serve:
        run_api()
    else:
        print("请指定运行模式：")
        print("训练模型: python main.py --train")
        print("启动API服务: python main.py --serve")