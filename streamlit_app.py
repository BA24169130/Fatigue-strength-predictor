import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1) 基本路径与配置文件读取
# =========================
# 当前 streamlit_app.py 所在目录
APP_DIR = Path(__file__).resolve().parent

# 读取模型配置文件
# 里面通常包含：
# - 模型文件名
# - scaler 文件名
# - 训练数据文件名
# - 特征名称
# - 特征显示标签
# - 输入范围
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
# 3) secrets 相关工具函数
# =========================
def get_secret_section(section_name: str) -> dict:
    """
    安全读取 st.secrets 的某个 section。
    如果没有配置，返回空字典，避免程序直接报错。
    """
    try:
        return dict(st.secrets[section_name])
    except Exception:
        return {}


def get_secret_value(section_name: str, key: str, default=None):
    """
    安全读取 st.secrets[section_name][key]。
    如果不存在，返回默认值。
    """
    try:
        return st.secrets[section_name][key]
    except Exception:
        return default


def normalize_email_set(values) -> set[str]:
    """
    把邮箱列表标准化成小写集合，便于后续判断权限。
    """
    if values is None:
        return set()

    return {
        str(v).strip().lower()
        for v in values
        if str(v).strip()
    }


def get_access_mode() -> str:
    """
    读取平台访问模式。

    支持三种模式：
    1. public -> 完全公开，所有人都能用
    2. login  -> 登录即可用
    3. paid   -> 登录后还要有授权才可用

    若 secrets 中没有配置，则默认给成 public，
    这样即使没配权限控制，平台也不会直接崩。
    """
    mode = str(get_secret_value("app_mode", "access_mode", "public")).strip().lower()
    if mode not in {"public", "login", "paid"}:
        mode = "public"
    return mode


def auth_is_configured() -> bool:
    """
    判断 Streamlit OIDC 登录是否已经配置完成。

    第一版按照单 provider 的最常见写法判断，
    至少要有：
    - redirect_uri
    - cookie_secret
    - client_id
    - client_secret
    - server_metadata_url
    """
    auth_section = get_secret_section("auth")
    if not auth_section:
        return False

    required_keys = [
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url",
    ]

    for key in required_keys:
        if not auth_section.get(key):
            return False

    return True


def safe_is_logged_in() -> bool:
    """
    安全判断当前用户是否已登录。

    注意：
    如果认证还没配置好，直接访问 st.user.is_logged_in
    可能会出现问题，所以这里先做保护。
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
    读取三类访问名单：
    - admin_users : 管理员
    - paid_users  : 已付费用户
    - free_users  : 你主动赠送权限的免费用户
    """
    access_section = get_secret_section("access_control")

    admin_users = normalize_email_set(access_section.get("admin_users", []))
    paid_users = normalize_email_set(access_section.get("paid_users", []))
    free_users = normalize_email_set(access_section.get("free_users", []))

    return admin_users, paid_users, free_users


def get_access_level(user_email: str) -> str:
    """
    判断当前用户的权限等级：
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
    读取购买链接，例如 Stripe Payment Link。
    """
    return str(get_secret_value("payments", "purchase_url", "")).strip()


def provider_name():
    """
    如果你后面想显式写 provider 名称，比如 google，
    可以返回 "google"。
    当前第一版默认直接 st.login()，所以返回 None。
    """
    return None


# =========================
# 4) 加载模型、scaler、训练数据
# =========================
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    加载模型、输入输出 scaler、训练数据。
    只有真正进入预测区时才会调用。
    """
    model = keras.models.load_model(APP_DIR / CONFIG["model_file"])
    scaler_x = joblib.load(APP_DIR / CONFIG["scaler_x_file"])
    scaler_y = joblib.load(APP_DIR / CONFIG["scaler_y_file"])
    train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
    return model, scaler_x, scaler_y, train_df


# =========================
# 5) 模型预测相关函数
# =========================
def predict_values(model, scaler_x, scaler_y, X_raw):
    """
    输入原始尺度 X_raw：
    1. scaler_x 标准化
    2. 模型预测
    3. scaler_y 反标准化
    """
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return y


def get_range_warnings(values):
    """
    检查输入是否超出训练数据范围。
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
    将 DataFrame 转为 Excel 二进制流，供下载按钮使用。
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


# =========================
# 6) 全局样式
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

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
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

    .mode-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #eef3ff;
        color: #26437a;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 10px;
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
# 7) 公开预览区
# =========================
def render_preview():
    """
    这个区域所有人都能看到。
    不管是未登录、未授权，还是正式用户，都会先看到这一块。
    """
    access_mode = get_access_mode()

    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">公开预览版：可查看平台介绍与使用说明；正式预测功能根据权限开放。</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="mode-badge">当前访问模式：{access_mode}</div>',
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
        <h3>使用说明</h3>
        <p>当前平台支持三种访问模式：</p>
        <p><b>public</b>：完全公开，所有人都能直接使用</p>
        <p><b>login</b>：登录即可使用</p>
        <p><b>paid</b>：只有已付费 / 免费授权 / 管理员才能使用</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 8) 登录 / 购买 / 未授权提示区
# =========================
def render_paywall(user_email: str):
    """
    这个函数只负责在不能使用正式预测功能时，给出对应提示。
    """
    purchase_url = get_purchase_url()
    mode = get_access_mode()

    st.markdown("---")
    st.subheader("访问控制说明")

    # 模式 1：public
    # 完全公开模式下，不应该进入这个函数
    # 但为了安全起见，还是做个兜底提示
    if mode == "public":
        st.info("当前模式为 public，理论上所有人都可直接使用预测功能。")
        return

    # 模式 2：login
    if mode == "login":
        if not auth_is_configured():
            st.warning("当前模式为 login，但登录认证尚未配置完成，因此暂时只能预览。")
            return

        if not safe_is_logged_in():
            st.warning("当前模式为 login。请先登录，登录后即可使用正式预测功能。")
            if st.button("使用 Google 登录", use_container_width=True):
                p = provider_name()
                if p:
                    st.login(p)
                else:
                    st.login()
            return

        # 已登录 normally 不会再走到这里
        st.info("你已登录，理论上应可直接使用预测功能。")
        return

    # 模式 3：paid
    if mode == "paid":
        # 认证未配置
        if not auth_is_configured():
            st.warning("当前模式为 paid，但登录认证尚未配置完成，因此暂时只能预览。")
            st.info("请先完成 OIDC 登录配置，再启用付费访问控制。")
            return

        # 未登录
        if not safe_is_logged_in():
            st.warning("当前为公开预览模式。请先登录，再购买或申请授权后使用正式预测功能。")
            c1, c2 = st.columns(2)

            with c1:
                if st.button("使用 Google 登录", use_container_width=True):
                    p = provider_name()
                    if p:
                        st.login(p)
                    else:
                        st.login()

            with c2:
                if purchase_url:
                    st.link_button("购买使用权", purchase_url, use_container_width=True)
                else:
                    st.button("购买使用权（请先配置 purchase_url）", disabled=True, use_container_width=True)
            return

        # 已登录但未授权
        st.warning(f"当前登录账号：{user_email}。你已登录，但尚未获得使用权限。")
        c1, c2 = st.columns(2)

        with c1:
            if purchase_url:
                st.link_button("立即购买", purchase_url, use_container_width=True)
            else:
                st.button("立即购买（请先配置 purchase_url）", disabled=True, use_container_width=True)

        with c2:
            if st.button("退出登录", use_container_width=True):
                st.logout()

        st.info("管理员说明：付款成功后，请将该邮箱加入 paid_users；若你要免费赠送权限，则加入 free_users。")
        return


# =========================
# 9) 管理员提示区
# =========================
def render_admin_panel():
    """
    管理员说明面板。
    第一版不做网页内自动修改名单，
    最稳的方式是直接在 secrets 中手动维护。
    """
    st.markdown("---")
    st.subheader("管理员说明")

    st.write("当前第一版采用手动白名单授权。")
    st.write("你可以在 Streamlit Community Cloud 的 Secrets 中维护：")

    st.code(
        """
[app_mode]
access_mode = "paid"

[access_control]
admin_users = ["你的邮箱@example.com"]
paid_users = ["已购买用户@example.com"]
free_users = ["免费授权用户@example.com"]
        """.strip()
    )

    st.write("说明：")
    st.write("- `access_mode = public`：所有人都能直接用")
    st.write("- `access_mode = login`：任何已登录用户都能用")
    st.write("- `access_mode = paid`：只有 admin / paid / free 才能用")

    st.write("白名单规则：")
    st.write("- `admin_users`：管理员，始终可用")
    st.write("- `paid_users`：已付费用户")
    st.write("- `free_users`：你主动赠送权限的用户")

    st.success("也就是说，你以后想免费开放时，只要把 access_mode 改成 login 或 public 即可；想单独赠送某些用户权限，就把他们加到 free_users。")


# =========================
# 10) 正式预测功能区
# =========================
def render_predictor_app():
    """
    这里保留你现在已经做好的正式预测平台。
    只有通过权限判断的用户才能进入。
    """
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

        train_df = pd.read_excel(APP_DIR / CONFIG["training_dataset_file"])
        range_df = pd.DataFrame({
            "Feature": CONFIG["feature_names"],
            "中文说明": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
            "Train min": [CONFIG["feature_ranges"][f]["train_min"] for f in CONFIG["feature_names"]],
            "Train max": [CONFIG["feature_ranges"][f]["train_max"] for f in CONFIG["feature_names"]],
            "Default": [CONFIG["feature_ranges"][f]["default"] for f in CONFIG["feature_names"]],
        })
        st.table(range_df)

        st.markdown("### 📌 使用建议")
        st.write("1. 尽量保证输入值位于训练数据范围内。")
        st.write("2. 若输入超出训练范围，结果只能作为参考。")
        st.write("3. 批量预测时请保持列名完全一致。")
        st.write("4. 当前版本直接调用你上传的原始模型文件。")


# =========================
# 11) 主入口逻辑
# =========================
def main():
    render_global_style()
    render_preview()

    mode = get_access_mode()

    # ========================================
    # 模式 A：public
    # 完全公开，所有人直接可用
    # ========================================
    if mode == "public":
        st.markdown("---")
        st.success("当前模式为 public：所有访问者都可直接使用正式预测功能。")
        render_predictor_app()
        return

    # ========================================
    # 模式 B：login
    # 只要登录就能用
    # ========================================
    if mode == "login":
        if not auth_is_configured():
            render_paywall(user_email="")
            return

        if not safe_is_logged_in():
            render_paywall(user_email="")
            return

        user_email = get_current_user_email()
        st.markdown("---")
        st.success(f"当前登录账号：{user_email}；模式：login")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("退出登录"):
                st.logout()
        with col2:
            st.empty()

        render_predictor_app()
        return

    # ========================================
    # 模式 C：paid
    # 只有 admin / paid / free 才能用
    # ========================================
    if mode == "paid":
        if not auth_is_configured():
            render_paywall(user_email="")
            return

        if not safe_is_logged_in():
            render_paywall(user_email="")
            return

        user_email = get_current_user_email()
        access_level = get_access_level(user_email)

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

        render_paywall(user_email=user_email)
        return


if __name__ == "__main__":
    main()
