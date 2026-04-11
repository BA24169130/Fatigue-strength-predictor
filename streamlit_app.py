import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1) 基本路径与配置读取
# =========================
# APP_DIR 表示当前 streamlit_app.py 所在目录
APP_DIR = Path(__file__).resolve().parent

# 读取配置文件，里面包含：
# - 模型文件名
# - scaler 文件名
# - 训练数据文件名
# - 特征名
# - 特征显示标签
# - 输入范围等
CONFIG = json.loads((APP_DIR / "model_config.json").read_text(encoding="utf-8"))


# =========================
# 2) 页面基础设置
# =========================
st.set_page_config(
    page_title="Fatigue Strength Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================
# 3) 加载模型、scaler、训练数据
# =========================
# 使用缓存，避免每次点击按钮都重复加载模型
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    加载：
    1. Keras 模型
    2. 输入标准化器 scaler_X
    3. 输出标准化器 scaler_y
    4. 训练数据（用于显示样本数、生成批量模板）
    """
    model = keras.models.load_model(APP_DIR / CONFIG["model_file"])
    scaler_x = joblib.load(APP_DIR / CONFIG["scaler_x_file"])
    scaler_y = joblib.load(APP_DIR / CONFIG["scaler_y_file"])
    train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
    return model, scaler_x, scaler_y, train_df


# =========================
# 4) 预测函数
# =========================
def predict_values(model, scaler_x, scaler_y, X_raw):
    """
    输入原始尺度的 X_raw
    -> scaler_x 标准化
    -> 模型预测
    -> scaler_y 反标准化
    最终得到原始尺度下的预测值
    """
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


# =========================
# 5) 检查是否超出训练范围
# =========================
def get_range_warnings(values):
    """
    检查输入值是否超出训练范围。
    若超出，则返回警告列表。
    """
    warnings = []

    for feat in CONFIG["feature_names"]:
        low = CONFIG["feature_ranges"][feat]["train_min"]
        high = CONFIG["feature_ranges"][feat]["train_max"]
        v = values[feat]

        if v < low or v > high:
            warnings.append(f"{feat} = {v:.4f} 超出训练范围 [{low:.4f}, {high:.4f}]")

    return warnings


# =========================
# 6) DataFrame 转 Excel 二进制流
# =========================
def df_to_xlsx_bytes(df):
    """
    用于下载按钮，把 DataFrame 导出为 Excel
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


# =========================
# 7) 主程序
# =========================
def main():
    model, scaler_x, scaler_y, train_df = load_assets()

    # =========================
    # 7.1 页面样式
    # =========================
    # 本次重点增强：
    # 1) “开始预测”按钮更高级
    # 2) 预测结果做成高亮结果卡片
    # 3) 其他样式尽量保持你当前版本
    st.markdown("""
    <style>
    /* ===== 整体基础字号 ===== */
    html, body, [class*="css"] {
        font-size: 18px;
    }

    /* ===== 主要内容区域顶部留白减小 ===== */
    .block-container {
        padding-top: 2rem !important;
    }

    /* ===== 主标题 ===== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0b3d91;
        margin-bottom: 0.3rem;
        text-align: center;
        letter-spacing: 0.5px;
    }

    /* ===== 副标题 ===== */
    .sub-title {
        font-size: 1.25rem;
        color: #444444;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    /* ===== 侧边栏紧凑化 ===== */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] .stWrite {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }

    [data-testid="stSidebar"] .stSubheader {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 18px !important;
    }

    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSidebar"] * {
        font-size: 16px !important;
    }

    /* ===== 标签页 ===== */
    button[data-baseweb="tab"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* ===== 小标题 ===== */
    h2, h3 {
        font-size: 1.8rem !important;
    }

    /* ===== 输入框标签 ===== */
    label, .stNumberInput label, .stTextInput label {
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* ===== 输入框里的数字 ===== */
    div[data-baseweb="input"] input {
        font-size: 22px !important;
    }

    /* ==================================================
       这里是“开始预测”按钮的高级样式
       以后若想继续改，就主要看这里
       ================================================== */
    .stButton > button {
        font-size: 28px !important;              /* 字体更大 */
        font-weight: 800 !important;             /* 更粗 */
        height: 3.8rem !important;               /* 按钮更高 */
        border-radius: 14px !important;          /* 圆角更明显 */
        border: none !important;
        color: white !important;
        background: linear-gradient(90deg, #ff4d4f 0%, #ff6b6b 100%) !important;
        box-shadow: 0 8px 18px rgba(255, 77, 79, 0.28) !important;
        transition: all 0.2s ease-in-out !important;
        letter-spacing: 0.5px !important;
    }

    /* 鼠标悬停时按钮更有交互感 */
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 22px rgba(255, 77, 79, 0.34) !important;
        background: linear-gradient(90deg, #ff4346 0%, #ff5c5c 100%) !important;
    }

    /* 按下时稍微回落 */
    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 6px 14px rgba(255, 77, 79, 0.25) !important;
    }

    /* ===== 普通提示框字体 ===== */
    [data-testid="stAlert"] {
        font-size: 20px !important;
    }

    /* ===== 表格列名（表头） ===== */
    [data-testid="stDataFrame"] thead th {
        font-size: 18px !important;
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    /* ===== 表格内数据 ===== */
    [data-testid="stDataFrame"] tbody td {
        font-size: 17px !important;
    }

    /* ==================================================
       这里是“预测结果”卡片的高级样式
       比普通 st.success 更醒目，也更像正式平台
       ================================================== */
    .prediction-card {
        margin-top: 14px;
        margin-bottom: 14px;
        padding: 18px 22px;
        border-radius: 16px;
        background: linear-gradient(135deg, #eefbf2 0%, #dff5e6 100%);
        border: 1.5px solid #b8e4c4;
        box-shadow: 0 8px 20px rgba(49, 130, 83, 0.12);
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .prediction-icon {
        width: 48px;
        height: 48px;
        min-width: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #20c997 0%, #198754 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(25, 135, 84, 0.28);
    }

    .prediction-text-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .prediction-label {
        font-size: 18px;
        color: #2c5f43;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .prediction-value {
        font-size: 32px;      /* 这里是结果主数字大小 */
        font-weight: 800;     /* 这里是结果主数字加粗 */
        color: #166534;
        line-height: 1.2;
    }

    .prediction-note {
        font-size: 16px;
        color: #4a6b58;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # 7.2 顶部标题
    # =========================
    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{CONFIG["subtitle_zh"]}</div>',
        unsafe_allow_html=True,
    )

    # =========================
    # 7.3 左侧侧边栏
    # =========================
    with st.sidebar:
        st.subheader("📁 已接入文件")
        st.write("✅ model.keras")
        st.write("✅ scaler_X.pkl")
        st.write("✅ scaler_y.pkl")
        st.write("✅ training_dataset.xlsx")
        st.markdown("---")

        st.subheader("📊 输入顺序")
        for i, feat in enumerate(CONFIG["feature_names"], start=1):
            st.write(f"{i}. {feat}")
        st.markdown("---")

        st.subheader("📈 训练数据规模")
        st.write(f"样本数：{len(train_df)}")
        st.write(f"特征数：{len(CONFIG['feature_names'])}")
        st.markdown("---")

        st.subheader("💡 说明")
        for note in CONFIG["notes"]:
            st.write(f"- {note}")

    # =========================
    # 7.4 三个标签页
    # =========================
    tab1, tab2, tab3 = st.tabs(["单点预测", "批量预测", "训练范围"])

    # =========================================================
    # Tab 1：单点预测
    # =========================================================
    with tab1:
        st.subheader("在线单点预测")
        col1, col2 = st.columns(2)
        values = {}

        # 根据配置文件自动生成输入框
        for i, feat in enumerate(CONFIG["feature_names"]):
            meta = CONFIG["feature_ranges"][feat]
            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                values[feat] = st.number_input(
                    label=CONFIG["feature_labels_zh"].get(feat, feat),
                    value=float(meta["default"]),
                    min_value=float(meta["ui_min"]),
                    max_value=float(meta["ui_max"]),
                    step=float(meta["step"]),
                    format="%.6f",
                )

        # 点击预测按钮
        if st.button("🔮 开始预测", type="primary", use_container_width=True):
            # 构造模型输入
            X = np.array([[values[f] for f in CONFIG["feature_names"]]], dtype=float)

            # 调用模型进行预测
            pred = float(predict_values(model, scaler_x, scaler_y, X)[0])

            # =====================================================
            # 这里不用 st.success，而是改成自定义“结果卡片”
            # 这样视觉上更高级，数字更突出
            # =====================================================
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-icon">✓</div>
                    <div class="prediction-text-group">
                        <div class="prediction-label">预测结果已生成</div>
                        <div class="prediction-value">{CONFIG['target_label_zh']} = {pred:.4f} MPa</div>
                        <div class="prediction-note">该结果由已部署的 FNN 模型实时计算得到</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 检查是否超出训练范围
            warnings = get_range_warnings(values)
            if warnings:
                st.warning("⚠️ 以下输入超出训练范围，当前结果属于外推，需谨慎解释：")
                for w in warnings:
                    st.write(f"- {w}")
            else:
                st.info("ℹ️ 当前输入处于训练数据范围内。")

            # 显示当前输入汇总
            st.dataframe(
                pd.DataFrame({
                    "Feature": CONFIG["feature_names"],
                    "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
                    "Value": [values[f] for f in CONFIG["feature_names"]],
                }),
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================
    # Tab 2：批量预测
    # =========================================================
    with tab2:
        st.subheader("批量预测")
        st.write("上传 xlsx 或 csv 文件，列名必须包含：E、σb、R、σ-1")
        uploaded = st.file_uploader("上传文件", type=["xlsx", "csv"])

        template_df = train_df[CONFIG["feature_names"]].head(10).copy()
        st.download_button(
            "📥 下载批量输入模板",
            data=df_to_xlsx_bytes(template_df),
            file_name="batch_input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if uploaded is not None:
            batch_df = (
                pd.read_csv(uploaded)
                if uploaded.name.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )

            st.write("上传数据预览")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing = [c for c in CONFIG["feature_names"] if c not in batch_df.columns]
            if missing:
                st.error(f"❌ 缺少必要列：{missing}")
            else:
                X_batch = batch_df[CONFIG["feature_names"]].astype(float).to_numpy()
                y_pred = predict_values(model, scaler_x, scaler_y, X_batch)

                out_df = batch_df.copy()
                out_df[CONFIG["target_label_zh"]] = y_pred

                st.success(f"✅ 已完成 {len(out_df)} 条数据预测。")
                st.dataframe(out_df.head(20), use_container_width=True)

                st.download_button(
                    "📥 下载预测结果",
                    data=df_to_xlsx_bytes(out_df),
                    file_name="fatigue_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # =========================================================
    # Tab 3：训练范围
    # =========================================================
    with tab3:
        st.subheader("训练数据范围")
        range_df = pd.DataFrame({
            "Feature": CONFIG["feature_names"],
            "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
            "Train min": [CONFIG["feature_ranges"][f]["train_min"] for f in CONFIG["feature_names"]],
            "Train max": [CONFIG["feature_ranges"][f]["train_max"] for f in CONFIG["feature_names"]],
            "Default": [CONFIG["feature_ranges"][f]["default"] for f in CONFIG["feature_names"]],
        })

        st.dataframe(range_df, use_container_width=True, hide_index=True)

        st.markdown("### 📌 使用建议")
        st.write("1. 尽量保证输入值位于训练数据范围内。")
        st.write("2. 若输入超出训练范围，结果只能作为参考。")
        st.write("3. 批量预测时请保持列名完全一致。")
        st.write("4. 当前版本直接调用你上传的原始模型文件。")


# =========================
# 8) 程序入口
# =========================
if __name__ == "__main__":
    main()
