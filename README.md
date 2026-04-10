# 金属材料应力比疲劳强度在线预测平台（正式版）

这个文件夹已经使用你提供的原始文件完成封装：
- model.keras
- scaler_X.pkl
- scaler_y.pkl
- training_dataset.xlsx
- FNN_training_script.py

## 功能
- 单点在线预测
- Excel / CSV 批量预测
- 训练范围检查
- Streamlit 与 Gradio 两个版本

## 推荐启动方式
### Streamlit
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Gradio
```bash
pip install -r requirements.txt
python gradio_app.py
```

## 说明
- 当前部署版会直接加载你上传的 model.keras 与两个 scaler 文件
- 输入顺序固定为：E、σb、R、σ-1
- 输出目标为：σa

## 建议上线方式
### 1. Streamlit Community Cloud
把本文件夹上传到 GitHub 仓库，然后部署 streamlit_app.py

### 2. Hugging Face Spaces
上传本文件夹，选择 Streamlit 或 Gradio SDK
