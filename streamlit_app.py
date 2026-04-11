import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

APP_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((APP_DIR / "model_config.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Fatigue Strength Predictor", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def load_assets():
    model = keras.models.load_model(APP_DIR / CONFIG["model_file"])
    scaler_x = joblib.load(APP_DIR / CONFIG["scaler_x_file"])
    scaler_y = joblib.load(APP_DIR / CONFIG["scaler_y_file"])
    train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
    return model, scaler_x, scaler_y, train_df


def predict_values(model, scaler_x, scaler_y, X_raw):
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


def get_range_warnings(values):
    warnings = []
    for feat in CONFIG["feature_names"]:
        low = CONFIG["feature_ranges"][feat]["train_min"]
        high = CONFIG["feature_ranges"][feat]["train_max"]
        v = values[feat]
        if v < low or v > high:
            warnings.append(f"{feat} = {v:.4f} 超出训练范围 [{low:.4f}, {high:.4f}]")
    return warnings


def df_to_xlsx_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


def main():
    model, scaler_x, scaler_y, train_df = load_assets()

    st.markdown("""
    <style>
    /* 整体基础字号 */
    html, body, [class*="css"] {
        font-size: 18px;
    }

    /* 主要内容区域：顶部留白（调整此值可改变顶部空白大小） */
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

    /* 侧边栏文字 */
    [data-testid="stSidebar"] * {
        font-size: 20px !important;
    }

    /* 标签页 */
    button[data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 600 !important;
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

    /* 按钮文字 */
    .stButton > button {
        font-size: 24px !important;
        font-weight: 700 !important;
        height: 3.2rem !important;
    }

    /* 成功/提示信息 */
    [data-testid="stAlert"] {
        font-size: 20px !important;
    }

    /* 表格文字 */
    [data-testid="stDataFrame"] div {
        font-size: 17px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{CONFIG["subtitle_zh"]}</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("已接入文件")
        st.write("✅ model.keras")
        st.write("✅ scaler_X.pkl")
        st.write("✅ scaler_y.pkl")
        st.write("✅ training_dataset.xlsx")
        st.markdown("---")

        st.subheader("输入顺序")
        for i, feat in enumerate(CONFIG["feature_names"], start=1):
            st.write(f"{i}. {feat}")
        st.markdown("---")

        st.subheader("训练数据规模")
        st.write(f"样本数：{len(train_df)}")
        st.write(f"特征数：{len(CONFIG['feature_names'])}")
        st.markdown("---")

        st.subheader("说明")
        for note in CONFIG["notes"]:
            st.write(f"- {note}")

    tab1, tab2, tab3 = st.tabs(["单点预测", "批量预测", "训练范围"])

    with tab1:
        st.subheader("在线单点预测")
        col1, col2 = st.columns(2)
        values = {}

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

        if st.button("开始预测", type="primary", use_container_width=True):
            X = np.array([[values[f] for f in CONFIG["feature_names"]]], dtype=float)
            pred = float(predict_values(model, scaler_x, scaler_y, X)[0])

            st.success(f"{CONFIG['target_label_zh']} = {pred:.4f}")

            warnings = get_range_warnings(values)
            if warnings:
                st.warning("以下输入超出训练范围，当前结果属于外推，需谨慎解释：")
                for w in warnings:
                    st.write(f"- {w}")
            else:
                st.info("当前输入处于训练数据范围内。")

            st.dataframe(
                pd.DataFrame({
                    "Feature": CONFIG["feature_names"],
                    "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
                    "Value": [values[f] for f in CONFIG["feature_names"]],
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.subheader("批量预测")
        st.write("上传 xlsx 或 csv 文件，列名必须包含：E、σb、R、σ-1")
        uploaded = st.file_uploader("上传文件", type=["xlsx", "csv"])

        template_df = train_df[CONFIG["feature_names"]].head(10).copy()
        st.download_button(
            "下载批量输入模板",
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
                st.error(f"缺少必要列：{missing}")
            else:
                X_batch = batch_df[CONFIG["feature_names"]].astype(float).to_numpy()
                y_pred = predict_values(model, scaler_x, scaler_y, X_batch)

                out_df = batch_df.copy()
                out_df[CONFIG["target_label_zh"]] = y_pred

                st.success(f"已完成 {len(out_df)} 条数据预测。")
                st.dataframe(out_df.head(20), use_container_width=True)

                st.download_button(
                    "下载预测结果",
                    data=df_to_xlsx_bytes(out_df),
                    file_name="fatigue_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

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

        st.markdown("### 使用建议")
        st.write("1. 尽量保证输入值位于训练数据范围内。")
        st.write("2. 若输入超出训练范围，结果只能作为参考。")
        st.write("3. 批量预测时请保持列名完全一致。")
        st.write("4. 当前版本直接调用你上传的原始模型文件。")


if __name__ == "__main__":
    main()
