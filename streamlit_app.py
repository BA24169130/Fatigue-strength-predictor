import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1) 基本路径与配置文件
# =========================
APP_DIR = Path(__file__).resolve().parent
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
# 3) 通用工具函数
# =========================
def get_secret_section(section_name: str) -> dict:
    """
    安全读取 st.secrets 的某个 section。
    若不存在，则返回空字典，避免直接报错。
    """
    try:
        return dict(st.secrets[section_name])
    except Exception:
        return {}


def get_secret_value(section_name: str, key: str, default=None):
    """
    安全读取 st.secrets[section_name][key]。
    若不存在，则返回 default。
    """
    try:
        return st.secrets[section_name][key]
    except Exception:
        return default


def normalize_email_set(values) -> set[str]:
    """
    把邮箱列表转成统一的小写集合，方便后续做权限判断。
    """
    if values is None:
        return set()
    return {
        str(v).strip().lower()
        for v in values
        if str(v).strip()
    }


def auth_is_configured() -> bool:
    """
    判断 Streamlit 的认证是否已配置完成。

    对于 OIDC，至少要有：
    - [auth]
    - redirect_uri
    - cookie_secret

    如果是单个默认 provider，还需要：
    - client_id
    - client_secret
    - server_metadata_url

    这里我们做“最小必要判断”。
    """
    auth_section = get_secret_section("auth")
    if not auth_section:
        return False

    required_shared_keys = ["redirect_uri", "cookie_secret"]
    for k in required_shared_keys:
        if not auth_section.get(k):
            return False

    # 单 provider 默认写法
    default_provider_keys = ["client_id", "client_secret", "server_metadata_url"]
    if all(auth_section.get(k) for k in default_provider_keys):
        return True

    # 多 provider 命名写法时，这里先简单认为 auth 已配置，
    # 具体 provider 名称由 st.login("provider_name") 使用
    # 第一版我们直接默认用 st.login()，因此建议先用单 provider。
    return False


def safe_is_logged_in() -> bool:
    """
    安全判断当前用户是否已登录。

    为什么不用直接 st.user.is_logged_in？
    因为如果认证还没正确配置，或者当前环境不支持，
    直接访问可能报 AttributeError。
    """
    if not auth_is_configured():
        return False

    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def get_current_user_email() -> str:
    """
    安全读取当前登录用户邮箱。
    若未登录或无法获取，则返回空字符串。
    """
    if not safe_is_logged_in():
        return ""

    try:
        email = st.user.get("email", "")
    except Exception:
        try:
            email = getattr(st.user, "email", "")
        except Exception:
            email = ""

    return str(email).strip().lower()


def get_access_lists():
    """
    从 secrets 中读取三类用户名单：
    - admin_users：管理员
    - paid_users：付费用户
    - free_users：免费授权用户
    """
    access_section = get_secret_section("access_control")

    admin_users = normalize_email_set(access_section.get("admin_users", []))
    paid_users = normalize_email_set(access_section.get("paid_users", []))
    free_users = normalize_email_set(access_section.get("free_users", []))

    return admin_users, paid_users, free_users


def get_access_level(user_email: str) -> str:
    """
    判断用户权限级别：
    - admin
    - paid
    - free
    - none
    """
    admin_users, paid_users, free_users = get_access_lists()

    if user_email in admin_users:
        return "admin"
    if user_email in paid_users:
        return "paid"
    if user_email in free_users:
        return "free"
    return "none"


def get_purchase_url() -> str:
    """
    读取购买链接（例如 Stripe Payment Link）。
    """
    return str(get_secret_value("payments", "purchase_url", "")).strip()


def provider_name() -> str | None:
    """
    第一版默认使用 unnamed/default provider。
    如果你后面改成命名 provider，可以在这里返回 provider 名称。
    """
    # 例如：return "google"
    return None


# =========================
# 4) 加载模型与训练资源
# =========================
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    加载模型、输入输出 scaler 和训练数据。
    """
    model = keras.models.load_model(APP_DIR / CONFIG["model_file"])
    scaler_x = joblib.load(APP_DIR / CONFIG["scaler_x_file"])
    scaler_y = joblib.load(APP_DIR / CONFIG["scaler_y_file"])
    train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
    return model, scaler_x, scaler_y, train_df


def predict_values(model, scaler_x, scaler_y, X_raw):
    """
    输入原始尺度的 X_raw：
    1) scaler_x 标准化
    2) 模型预测
    3) scaler_y 反标准化
    """
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


def get_range_warnings(values):
    """
    检查输入值是否超出训练范围。
    """
    warnings = []
    for feat in CONFIG["feature_names"]:
        low = CONFIG["feature_ranges"][feat]["train_min"]
        high = CONFIG["feature_ranges"][feat]["train_max"]
        v = values[feat]
        if v < low or v > high:
            warnings.append(f"{feat} = {v:.4f} 超出训练范围 [{low:.4f}, {high:.4f}]")
    return warnings


def df_to_xlsx_bytes(df):
    """
    将 DataFrame 导出为 Excel 二进制流，用于下载。
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


# =========================
# 5) 页面样式
# =========================
def render_global_style():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0b3d91;
        margin-bottom: 0.3rem;
        text-align: center;
    }

    .sub-title {
        font-size: 1.25rem;
        color: #444444;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    [data-testid="stSidebar"] * {
        font-size: 16px !important;
    }

    button[data-baseweb="tab"] {
        padding: 10px 18px !important;
        min-height: 48px !important;
    }

    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        font-size: 1.9rem !important;
    }

    label, .stNumberInput label, .stTextInput label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"] input {
        font-size: 21px !important;
    }

    .stButton > button {
        font-size: 22px !important;
        font-weight: 700 !important;
        height: 3.2rem !important;
    }

    [data-testid="stAlert"] {
        font-size: 19px !important;
    }

    .prediction-result {
        background-color: #dff0df;
        color: #1e7a46;
        border-radius: 0.5rem;
        padding: 16px 18px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.4;
    }

    .preview-card {
        border: 1px solid #d9e3f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 12px;
        background: #f8fbff;
    }

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
# 6) 公开预览页
# =========================
def render_preview():
    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">公开预览版：可查看平台介绍与使用说明；正式预测需登录并获得授权。</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="preview-card">
        <h3>平台简介</h3>
        <p>本平台用于金属材料应力比疲劳强度在线预测。</p>
        <p><b>公开可见：</b>平台介绍、输入说明、适用范围、使用流程。</p>
        <p><b>正式功能：</b>单点预测、批量预测、结果下载。</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="preview-card">
        <h3>输入特征</h3>
        <p>1. E：弹性模量</p>
        <p>2. σb：抗拉强度</p>
        <p>3. R：应力比</p>
        <p>4. σ-1：R = -1 时疲劳强度</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="preview-card">
        <h3>使用方式</h3>
        <p>步骤 1：登录平台</p>
        <p>步骤 2：购买使用权</p>
        <p>步骤 3：管理员开通后即可使用</p>
        <p>第一版采用手动白名单授权。</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 7) 未登录 / 未授权时的提示区
# =========================
def render_paywall(user_email: str, access_level: str):
    purchase_url = get_purchase_url()

    st.markdown("---")
    st.subheader("正式使用权限")

    # 认证未配置
    if not auth_is_configured():
        st.warning("当前平台还未完成登录认证配置，因此暂时处于仅预览模式。")
        st.info("下一步请先配置 Streamlit OIDC 认证，再启用付费访问控制。")
        return

    # 未登录
    if not safe_is_logged_in():
        st.warning("当前为公开预览模式。请先登录，再购买或申请授权后使用预测功能。")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("使用 Google 登录", use_container_width=True):
                p = provider_name()
                if p:
                    st.login(p)
                else:
                    st.login()

        with col2:
            if purchase_url:
                st.link_button("购买使用权", purchase_url, use_container_width=True)
            else:
                st.button("购买使用权（请先配置 purchase_url）", disabled=True, use_container_width=True)
        return

    # 已登录但未授权
    st.warning(f"当前登录账号：{user_email}。你已登录，但尚未获得使用权限。")

    col1, col2 = st.columns(2)
    with col1:
        if purchase_url:
            st.link_button("立即购买", purchase_url, use_container_width=True)
        else:
            st.button("立即购买（请先配置 purchase_url）", disabled=True, use_container_width=True)

    with col2:
        if st.button("退出登录", use_container_width=True):
            st.logout()

    st.info("管理员说明：付款成功后，请将该邮箱加入 paid_users；若你要免费赠送权限，则加入 free_users。")


# =========================
# 8) 管理员说明区
# =========================
def render_admin_panel():
    st.markdown("---")
    st.subheader("管理员说明")

    st.write("第一版采用手动白名单授权。")
    st.write("你可以在 Streamlit Community Cloud 的 Secrets 中维护：")
    st.code(
        """
[access_control]
admin_users = ["你的邮箱@example.com"]
paid_users = ["已购买用户@example.com"]
free_users = ["免费授权用户@example.com"]
        """.strip()
    )

    st.write("规则如下：")
    st.write("- `admin_users`：管理员，始终可用")
    st.write("- `paid_users`：已购买用户")
    st.write("- `free_users`：你主动赠送权限的免费用户")

    st.success("也就是说，你要给没买过的人免费权限时，只要把他的邮箱加入 free_users。")


# =========================
# 9) 正式预测区
# =========================
def render_predictor_app():
    model, scaler_x, scaler_y, train_df = load_assets()

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

    st.success("你已获得使用权限，下面是正式预测功能。")

    tab1, tab2, tab3 = st.tabs(["单点预测", "批量预测", "训练范围"])

    with tab1:
        st.subheader("单点预测")
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

        if st.button("🔮 开始预测", type="primary", use_container_width=True):
            X = np.array([[values[f] for f in CONFIG["feature_names"]]], dtype=float)
            pred = float(predict_values(model, scaler_x, scaler_y, X)[0])

            st.markdown(
                f"""
                <div class="prediction-result">
                    ✅ {CONFIG['target_label_zh']} = {pred:.4f} MPa
                </div>
                """,
                unsafe_allow_html=True,
            )

            warnings = get_range_warnings(values)
            if warnings:
                st.warning("⚠️ 以下输入超出训练范围，当前结果属于外推，需谨慎解释：")
                for w in warnings:
                    st.write(f"- {w}")
            else:
                st.info("ℹ️ 当前输入处于训练数据范围内。")

            result_df = pd.DataFrame({
                "Feature": CONFIG["feature_names"],
                "中文说明": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
                "Value": [values[f] for f in CONFIG["feature_names"]],
            })
            st.table(result_df)

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
            if uploaded.name.lower().endswith(".csv"):
                batch_df = pd.read_csv(uploaded)
            else:
                batch_df = pd.read_excel(uploaded)

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

    with tab3:
        st.subheader("训练范围")
        range_df = pd.DataFrame({
            "Feature": CONFIG["feature_names"],
            "中文说明": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
            "Train min": [CONFIG["feature_ranges"][f]["train_min"] for f in CONFIG["feature_names"]],
            "Train max": [CONFIG["feature_ranges"][f]["train_max"] for f in CONFIG["feature_names"]],
            "Default": [CONFIG["feature_ranges"][f]["default"] for f in CONFIG["feature_names"]],
        })
        st.table(range_df)


# =========================
# 10) 主入口
# =========================
def main():
    render_global_style()
    render_preview()

    # 认证未配置时：安全退回到预览模式，避免直接报错
    if not auth_is_configured():
        render_paywall(user_email="", access_level="none")
        return

    # 未登录：只显示预览与登录/购买按钮
    if not safe_is_logged_in():
        render_paywall(user_email="", access_level="none")
        return

    # 已登录：判断权限
    user_email = get_current_user_email()
    access_level = get_access_level(user_email)

    # 管理员、付费用户、免费授权用户：允许进入正式预测
    if access_level in {"admin", "paid", "free"}:
        st.markdown("---")
        st.success(f"当前登录账号：{user_email}；权限：{access_level}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("退出登录"):
                st.logout()
        with col2:
            st.empty()

        if access_level == "admin":
            render_admin_panel()

        render_predictor_app()
        return

    # 已登录但未授权
    render_paywall(user_email=user_email, access_level=access_level)


if __name__ == "__main__":
    main()
