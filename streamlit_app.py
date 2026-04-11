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
# 3. secrets 读取与访问控制工具函数
# =========================
def get_secret_section(section_name: str) -> dict:
    """
    安全读取 st.secrets 的某个 section。
    如果没有配置，就返回空字典，避免程序直接报错。
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
    把邮箱列表统一处理为小写集合，便于做权限判断。
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
    1) public -> 完全公开，所有人可直接使用
    2) login  -> 登录即可使用
    3) paid   -> 登录后还需要在权限名单里才可使用

    如果没配置，就默认用 public，保证平台至少可以工作。
    """
    mode = str(get_secret_value("app_mode", "access_mode", "public")).strip().lower()
    if mode not in {"public", "login", "paid"}:
        mode = "public"
    return mode


def auth_is_configured() -> bool:
    """
    判断 Streamlit OIDC 登录是否已经正确配置。

    第一版先按最常见的单 provider 方式判断，
    至少需要：
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

    之所以不直接写 st.user.is_logged_in，
    是因为如果认证还没配置好，直接访问可能出错。
    """
    if not auth_is_configured():
        return False

    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def get_current_user_email() -> str:
    """
    安全获取当前登录用户邮箱。
    如果没有登录，则返回空字符串。
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
    - free_users  : 免费授权用户（你主动赠送权限的人）
    """
    access_section = get_secret_section("access_control")

    admin_users = normalize_email_set(access_section.get("admin_users", []))
    paid_users = normalize_email_set(access_section.get("paid_users", []))
    free_users = normalize_email_set(access_section.get("free_users", []))

    return admin_users, paid_users, free_users


def get_access_level(user_email: str) -> str:
    """
    根据邮箱判断当前用户的权限等级：
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


# =========================
# 4. 加载模型、scaler 和训练数据
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
# 5. 预测函数
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
# 6. 输入范围检查
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
# 7. DataFrame 导出为 Excel 字节流
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
# 8. 全局样式
# =========================
def render_global_style():
    """
    尽量保持你原来平台的视觉风格不变。
    这里只在你当前 CSS 基础上，增加了少量访问控制模块需要的样式。
    """
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
        font-size: 30px;
        font-weight: 800;
        line-height: 1.4;
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

    /* ===== 访问控制预览卡片 ===== */
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
    </style>
    """, unsafe_allow_html=True)


# =========================
# 9. 预览区（所有人都能看）
# =========================
def render_preview_section():
    """
    平台公开预览区。
    无论是否登录、是否付费，所有人都能看到。
    """
    mode = get_access_mode()

    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{CONFIG["subtitle_zh"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="mode-badge">当前访问模式：{mode}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="preview-card">
        <h3>平台简介（公开预览）</h3>
        <p>本平台用于金属材料应力比疲劳强度在线预测。</p>
        <p>当前所有访问者都可查看平台介绍、输入说明和使用方式。</p>
        <p>正式预测功能是否开放，取决于平台访问模式与当前用户权限。</p>
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
        <h3>访问模式说明</h3>
        <p><b>public</b>：所有人都能直接使用预测功能</p>
        <p><b>login</b>：登录后即可使用预测功能</p>
        <p><b>paid</b>：只有管理员、已付费用户、免费授权用户才能使用预测功能</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 10. 管理员说明区
# =========================
def render_admin_panel():
    """
    给管理员看的说明区。
    第一版不做网页内自动改名单，
    最稳的方式还是在 secrets 中手动维护名单。
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
    st.write("- `free_users`：你主动赠送免费权限的用户名单")

    st.success("如果你想给一个没买过的人免费权限，只需要把他的邮箱加入 free_users。")


# =========================
# 11. 登录 / 购买 / 未授权提示区
# =========================
def render_access_prompt(user_email: str = ""):
    """
    当用户尚不能使用正式预测功能时，
    在预览区下面给出相应的操作提示。
    """
    mode = get_access_mode()
    purchase_url = get_purchase_url()

    st.markdown("---")
    st.subheader("正式使用入口")

    # public 模式理论上不应该走到这里
    if mode == "public":
        st.info("当前模式为 public，所有用户都可直接使用预测功能。")
        return

    # login 模式：只要登录就可以
    if mode == "login":
        if not auth_is_configured():
            st.warning("当前模式为 login，但登录认证尚未配置完成，因此暂时只能预览。")
            st.info("请先完成 OIDC 登录配置，然后平台即可切换为“登录即可使用”。")
            return

        if not safe_is_logged_in():
            st.warning("当前模式为 login。请先登录，登录后即可使用预测功能。")
            if st.button("使用 Google 登录", use_container_width=True):
                st.login()
            return

        st.info("你已登录，理论上应可直接使用预测功能。")
        return

    # paid 模式：登录 + 已授权
    if mode == "paid":
        if not auth_is_configured():
            st.warning("当前模式为 paid，但登录认证尚未配置完成，因此暂时只能预览。")
            st.info("请先完成 OIDC 登录配置，再启用付费访问控制。")
            return

        if not safe_is_logged_in():
            st.warning("当前为公开预览模式。请先登录，再购买或申请授权后使用正式预测功能。")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("使用 Google 登录", use_container_width=True):
                    st.login()

            with col2:
                if purchase_url:
                    st.link_button("购买使用权", purchase_url, use_container_width=True)
                else:
                    st.button("购买使用权（请先配置 purchase_url）", disabled=True, use_container_width=True)
            return

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

        st.info("管理员说明：付款成功后，请将该邮箱加入 paid_users；若你想免费赠送权限，则加入 free_users。")
        return


# =========================
# 12. 正式预测平台（尽量保留你原有布局）
# =========================
def render_predictor_app():
    """
    这里尽量完整保留你原来的平台布局与内容，
    只是在外层增加访问控制。
    """
    model, scaler_x, scaler_y, train_df = load_assets()

    # =========================
    # 左侧侧边栏
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
    # 主体标签页
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

            # 使用自定义 prediction-result 样式来显示结果
            st.markdown(
                f"""
                <div class="prediction-result">
                    ✅ {CONFIG['target_label_zh']} = {pred:.4f} MPa
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

            # 结果汇总表
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
# 13. 主程序入口
# =========================
def main():
    """
    主入口逻辑：
    1) 永远先显示公开预览区
    2) 根据 access_mode 决定后续是否开放正式预测功能
    """
    render_global_style()
    render_preview_section()

    mode = get_access_mode()

    # ========================================
    # 模式 A：public
    # 完全公开，所有人都能直接使用
    # ========================================
    if mode == "public":
        st.markdown("---")
        st.success("当前模式为 public：所有访问者都可直接使用正式预测功能。")
        render_predictor_app()
        return

    # ========================================
    # 模式 B：login
    # 登录即可使用
    # ========================================
    if mode == "login":
        if not auth_is_configured():
            render_access_prompt()
            return

        if not safe_is_logged_in():
            render_access_prompt()
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
    # 只有 admin / paid / free 才能使用
    # ========================================
    if mode == "paid":
        if not auth_is_configured():
            render_access_prompt()
            return

        if not safe_is_logged_in():
            render_access_prompt()
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

        render_access_prompt(user_email=user_email)
        return


# =========================
# 14. 程序入口
# =========================
if __name__ == "__main__":
    main()
