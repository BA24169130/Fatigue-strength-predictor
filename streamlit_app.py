import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1. 基本路径与配置文件读取
# =========================
# APP_DIR 表示当前这个 streamlit_app.py 所在的文件夹
APP_DIR = Path(__file__).resolve().parent

# 读取模型配置文件
# 这个 json 里通常包含：
# - 模型文件名
# - scaler 文件名
# - 训练数据文件名
# - 特征名称
# - 特征显示标签
# - 输入范围等
CONFIG = json.loads((APP_DIR / "model_config.json").read_text(encoding="utf-8"))


# =========================
# 2. 页面基础设置
# =========================
st.set_page_config(
    page_title="Fatigue Strength Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================
# 3. 加载模型、scaler 和训练数据
# =========================
# 使用缓存的好处：
# 页面每次交互时不会重复加载模型，速度会更快
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    加载：
    1) Keras 模型
    2) 输入特征标准化器 scaler_X
    3) 输出标准化器 scaler_y
    4) 训练数据集（用于展示样本数、生成模板等）
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
    输入原始尺度的 X_raw
    先用 scaler_x 标准化，再用模型预测，
    最后再用 scaler_y 反标准化，得到原始尺度预测值
    """
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


# =========================
# 5. 输入范围检查
# =========================
def get_range_warnings(values):
    """
    检查当前输入是否超出训练范围。
    如果超出，则返回警告信息列表。
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
# 6. DataFrame 导出为 Excel 字节流
# =========================
def df_to_xlsx_bytes(df):
    """
    将 DataFrame 导出成 Excel 的二进制内容，
    方便 streamlit 的下载按钮直接下载
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
    # 7.1 页面样式（CSS）
    # =========================
    # 这里是本次重点修改部分：
    # 1) 把标签页字体调大
    # 2) 把输入框、按钮、侧边栏、提示框字体调大
    # 3) 给 st.table 的表格字体单独放大
    st.markdown("""
    <style>
    /* ===== 页面整体基础字号 ===== */
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
    }

    /* ===== 副标题 ===== */
    .sub-title {
        font-size: 1.3rem;
        color: #444444;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    /* ===== 侧边栏整体紧凑一些 ===== */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] .stWrite {
        margin-bottom: 0.2rem !important;
        line-height: 1.35 !important;
    }

    [data-testid="stSidebar"] .stSubheader {
        margin-top: 0.55rem !important;
        margin-bottom: 0.35rem !important;
        font-size: 18px !important;
    }

    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 侧边栏文字稍微加大 */
    [data-testid="stSidebar"] * {
        font-size: 17px !important;
    }

    /* ===== 标签页外观 ===== */
    button[data-baseweb="tab"] {
        padding: 10px 18px !important;
        min-height: 48px !important;
    }

    /* 这是你截图里上面那排“单点预测 / 批量预测 / 训练范围”文字大小 */
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    /* ===== 小标题，比如“单点预测” ===== */
    h2, h3 {
        font-size: 1.9rem !important;
    }

    /* ===== 输入框标签 ===== */
    label, .stNumberInput label, .stTextInput label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* ===== 输入框里的数字 ===== */
    div[data-baseweb="input"] input {
        font-size: 21px !important;
    }

    /* ===== 按钮文字 ===== */
    .stButton > button {
        font-size: 22px !important;
        font-weight: 700 !important;
        height: 3.2rem !important;
    }

    /* ===== 成功、警告、提示框 ===== */
    [data-testid="stAlert"] {
        font-size: 19px !important;
    }

    /* ===== st.table 表格字体 ===== */
    [data-testid="stTable"] table {
        font-size: 19px !important;
        width: 100% !important;
    }

    [data-testid="stTable"] thead tr th {
        font-size: 20px !important;
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    [data-testid="stTable"] tbody tr td {
        font-size: 19px !important;
    }

    /* ===== 如果页面里还有 st.dataframe，也尽量放大一点 ===== */
    [data-testid="stDataFrame"] thead th {
        font-size: 18px !important;
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    [data-testid="stDataFrame"] tbody td {
        font-size: 17px !important;
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
    # 7.4 主体标签页
    # =========================
    tab1, tab2, tab3 = st.tabs(["单点预测", "批量预测", "训练范围"])

    # =========================================================
    # Tab 1：单点预测
    # =========================================================
    with tab1:
        st.subheader("单点预测")

        # 两列布局
        col1, col2 = st.columns(2)
        values = {}

        # 根据配置文件自动生成输入框
        for i, feat in enumerate(CONFIG["feature_names"]):
            meta = CONFIG["feature_ranges"][feat]

            # 偶数特征放左边，奇数特征放右边
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

            # 调用预测函数
            pred = float(predict_values(model, scaler_x, scaler_y, X)[0])

            # 显示预测结果
            st.success(f"✅ {CONFIG['target_label_zh']} = {pred:.4f} MPa")

            # 检查是否超出训练范围
            warnings = get_range_warnings(values)
            if warnings:
                st.warning("⚠️ 以下输入超出训练范围，当前结果属于外推，需谨慎解释：")
                for w in warnings:
                    st.write(f"- {w}")
            else:
                st.info("ℹ️ 当前输入处于训练数据范围内。")

            # ===== 这里是你截图里下方那张小表 =====
            # 原来你用的是 st.dataframe，字体偏小且不容易精细控制
            # 这里改成 st.table，更适合展示少量汇总信息，而且字体更容易放大
            result_df = pd.DataFrame({
                "Feature": CONFIG["feature_names"],
                "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
                "Value": [values[f] for f in CONFIG["feature_names"]],
            })
            st.table(result_df)

    # =========================================================
    # Tab 2：批量预测
    # =========================================================
    with tab2:
        st.subheader("批量预测")
        st.write("上传 xlsx 或 csv 文件，列名必须包含：E、σb、R、σ-1")

        uploaded = st.file_uploader("上传文件", type=["xlsx", "csv"])

        # 生成一个批量输入模板，默认取训练集前10行对应的输入特征
        template_df = train_df[CONFIG["feature_names"]].head(10).copy()

        st.download_button(
            "📥 下载批量输入模板",
            data=df_to_xlsx_bytes(template_df),
            file_name="batch_input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if uploaded is not None:
            # 自动识别 csv 或 xlsx
            if uploaded.name.lower().endswith(".csv"):
                batch_df = pd.read_csv(uploaded)
            else:
                batch_df = pd.read_excel(uploaded)

            st.write("上传数据预览")
            st.dataframe(batch_df.head(), use_container_width=True)

            # 检查列名是否齐全
            missing = [c for c in CONFIG["feature_names"] if c not in batch_df.columns]
            if missing:
                st.error(f"❌ 缺少必要列：{missing}")
            else:
                # 批量预测
                X_batch = batch_df[CONFIG["feature_names"]].astype(float).to_numpy()
                y_pred = predict_values(model, scaler_x, scaler_y, X_batch)

                # 结果拼回原表
                out_df = batch_df.copy()
                out_df[CONFIG["target_label_zh"]] = y_pred

                st.success(f"✅ 已完成 {len(out_df)} 条数据预测。")
                st.dataframe(out_df.head(20), use_container_width=True)

                # 下载预测结果
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

        # 这里也改成 st.table，让“训练范围”这一页的表格字体更大、更清楚
        st.table(range_df)

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
