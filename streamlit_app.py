import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1. 读取路径与配置文件
# =========================
# APP_DIR 表示当前 streamlit_app.py 所在文件夹
APP_DIR = Path(__file__).resolve().parent

# 读取配置文件 model_config.json
CONFIG = json.loads((APP_DIR / "model_config.json").read_text(encoding="utf-8"))


# =========================
# 2. 设置网页基础信息
# =========================
st.set_page_config(
    page_title="Fatigue Strength Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================
# 3. 加载模型、scaler 和训练数据
# =========================
# 使用缓存，避免每次点击按钮都重复加载模型，提高速度
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    加载：
    1) 训练好的 Keras 模型
    2) 输入特征标准化器 scaler_X
    3) 输出标准化器 scaler_y
    4) 训练数据集（用于显示样本数、生成批量模板）
    """
    model = keras.models.load_model(APP_DIR / CONFIG["model_file"])
    scaler_x = joblib.load(APP_DIR / CONFIG["scaler_x_file"])
    scaler_y = joblib.load(APP_DIR / CONFIG["scaler_y_file"])
    train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
    return model, scaler_x, scaler_y, train_df


# =========================
# 4. 预测函数
# =========================
def predict_values(model, scaler_x, scaler_y, X_raw):
    """
    输入原始尺度 X_raw
    -> 用 scaler_x 标准化
    -> 用模型预测
    -> 用 scaler_y 反标准化
    最终返回原始尺度的预测值
    """
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


# =========================
# 5. 检查输入是否超出训练范围
# =========================
def get_range_warnings(values):
    """
    如果输入超出训练数据范围，则给出警告信息
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
# 6. DataFrame 转为 Excel 下载流
# =========================
def df_to_xlsx_bytes(df):
    """
    将 DataFrame 转成 Excel 二进制，方便下载按钮下载
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


# =========================
# 7. 主程序
# =========================
def main():
    # 先加载模型和数据
    model, scaler_x, scaler_y, train_df = load_assets()

    # =========================
    # 7.1 页面样式 CSS
    # =========================
    # 你这次最关心的是两处：
    # 1) “开始预测”按钮字体变大变粗
    # 2) 预测结果那一行字体变大变粗
    # 所以我专门新增了 .prediction-result 样式，
    # 并把 .stButton > button 的字号和字重调大
    st.markdown("""
    <style>
    /* 整体基础字号 */
    html, body, [class*="css"] {
        font-size: 18px;
    }

    /* 主要内容区域：顶部留白减小 */
    .block-container {
        padding-top: 2rem !important;
    }

    /* 主标题：居中 + 样式 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0b3d91;
        margin-bottom: 0.3rem;
        text-align: center;
    }

    /* 副标题：居中 + 样式 */
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

    /* 标签页 */
    button[data-baseweb="tab"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* 小标题 */
    h2, h3 {
        font-size: 1.8rem !important;
    }

    /* 输入框标签 */
    label, .stNumberInput label, .stTextInput label {
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* 输入框里的数字 */
    div[data-baseweb="input"] input {
        font-size: 22px !important;
    }

    /* =========================
       这里是“开始预测”按钮的样式
       如果你以后还想继续放大，就改这里
       ========================= */
    .stButton > button {
        font-size: 28px !important;     /* 原来更小，这里调大 */
        font-weight: 800 !important;    /* 加粗 */
        height: 3.6rem !important;      /* 按钮高度也略增大 */
    }

    /* 普通提示框字体 */
    [data-testid="stAlert"] {
        font-size: 20px !important;
    }

    /* 表格列名（表头）样式 */
    [data-testid="stDataFrame"] thead th {
        font-size: 18px !important;
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    /* 表格内数据 */
    [data-testid="stDataFrame"] tbody td {
        font-size: 17px !important;
    }

    /* =========================
       这里是预测结果文字的专属样式
       只影响“预测疲劳强度 ...”这一行
       ========================= */
    .prediction-result {
        background-color: #dff0df;
        color: #1e7a46;
        border-radius: 0.5rem;
        padding: 16px 18px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        font-size: 30px;      /* 结果字体放大 */
        font-weight: 800;     /* 结果加粗 */
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # 7.2 页面顶部标题
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

        # 自动生成输入框
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
            # 组装模型输入
            X = np.array([[values[f] for f in CONFIG["feature_names"]]], dtype=float)

            # 调用模型预测
            pred = float(predict_values(model, scaler_x, scaler_y, X)[0])

            # =========================
            # 这里不再用 st.success
            # 因为你只想把“预测结果”这一行单独放大加粗
            # 所以改成自定义的 HTML 样式框
            # =========================
            st.markdown(
                f"""
                <div class="prediction-result">
                    ✅ {CONFIG['target_label_zh']} = {pred:.4f} MPa
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 显示范围检查结果
            warnings = get_range_warnings(values)
            if warnings:
                st.warning("⚠️ 以下输入超出训练范围，当前结果属于外推，需谨慎解释：")
                for w in warnings:
                    st.write(f"- {w}")
            else:
                st.info("ℹ️ 当前输入处于训练数据范围内。")

            # 显示当前输入信息
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

        # 批量预测模板
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
# 8. 程序入口
# =========================
if __name__ == "__main__":
    main()
