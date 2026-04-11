import io
import json
import sqlite3
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# =========================
# 1. 基本路径与配置文件读取
# =========================
APP_DIR = Path(__file__).resolve().parent

# 读取模型配置文件
CONFIG = json.loads((APP_DIR / "model_config.json").read_text(encoding="utf-8"))

# SQLite 数据库文件
VISITOR_DB_FILE = Path("/tmp") / "visitor_counter.db"


# =========================
# 2. 页面基础设置
# =========================
st.set_page_config(
    page_title="Fatigue Strength Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================
# 3. secrets 与访问控制辅助函数
# =========================
def get_secret_section(section_name: str) -> dict:
    """
    安全读取 st.secrets 的某个 section。
    若不存在，则返回空字典，避免程序直接报错。
    """
    try:
        return dict(st.secrets[section_name])
    except Exception:
        return {}


def get_secret_value(section_name: str, key: str, default=None):
    """
    安全读取 st.secrets[section_name][key]。
    若不存在，则返回默认值。
    """
    try:
        return st.secrets[section_name][key]
    except Exception:
        return default


def normalize_email_set(values) -> set:
    """
    把邮箱列表统一转换成小写集合，便于后续判断权限。
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
    当前只保留两种模式：
    1) public      -> 免费公开版本
    2) authorized  -> 通过授权后才能使用

    如果没有配置，则默认 public。
    """
    mode = str(get_secret_value("app_mode", "access_mode", "public")).strip().lower()
    if mode not in {"public", "authorized"}:
        mode = "public"
    return mode


def auth_is_configured() -> bool:
    """
    判断 OIDC 登录是否已经配置完成。
    这里按最常见的单 provider 配置判断。
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
    避免在未配置 auth 时直接访问 st.user.is_logged_in 导致报错。
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
    读取权限名单：
    - admin_users：管理员
    - authorized_users：已授权用户
    """
    access_section = get_secret_section("access_control")

    admin_users = normalize_email_set(access_section.get("admin_users", []))
    authorized_users = normalize_email_set(access_section.get("authorized_users", []))

    return admin_users, authorized_users


def is_authorized_user(user_email: str) -> bool:
    """
    判断某个用户是否具备正式使用权限。
    """
    admin_users, authorized_users = get_access_lists()
    return user_email in admin_users or user_email in authorized_users


# =========================
# 4. SQLite 数据库版访问人数统计
# =========================
@st.cache_resource
def get_db_lock():
    """
    全局线程锁。
    用于避免多人同时访问时写数据库冲突。
    """
    return threading.Lock()


def get_db_connection():
    """
    创建 SQLite 连接。
    每次操作都单独打开一个连接，避免跨线程共用连接。
    增加 timeout，提高并发时的稳定性。
    """
    conn = sqlite3.connect(VISITOR_DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def rebuild_visits_table(conn):
    """
    当旧表结构不兼容时，重建 visits 表。
    """
    conn.execute("DROP TABLE IF EXISTS visits")
    conn.execute(
        """
        CREATE TABLE visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL UNIQUE,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def init_visitor_db():
    """
    初始化访问统计数据库。
    """
    with get_db_lock():
        conn = get_db_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key TEXT NOT NULL UNIQUE,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

            cur = conn.execute("PRAGMA table_info(visits)")
            columns = [row[1] for row in cur.fetchall()]

            if "session_key" not in columns:
                rebuild_visits_table(conn)

        finally:
            conn.close()


def get_or_create_visit_key() -> str:
    """
    获取当前页面访问标识。

    逻辑：
    1) 如果 URL 里已经有 v 参数，说明这是同一次进入后的刷新或交互，不重复计数
    2) 如果没有 v 参数，说明是一次新的进入，生成新的 visit_key，并写入 URL
    """
    params = st.query_params

    if "v" in params and str(params["v"]).strip():
        return str(params["v"]).strip()

    new_key = str(uuid.uuid4())
    st.query_params["v"] = new_key
    return new_key


def register_visit_once_per_entry() -> int:
    """
    每次真正“进入页面”只累计一次。
    同一个页面刷新时，因为 URL 中保留同一个 visit_key，所以不会重复计数。
    """
    init_visitor_db()
    visit_key = get_or_create_visit_key()

    with get_db_lock():
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO visits (session_key) VALUES (?)",
                (visit_key,)
            )
            conn.commit()

            cur = conn.execute("SELECT COUNT(*) FROM visits")
            total_visits = int(cur.fetchone()[0])
        finally:
            conn.close()

    return total_visits


def render_visitor_counter():
    """
    在标题下方显示累计访问人数。
    """
    total_visits = register_visit_once_per_entry()

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:center;
            margin-top:-2px;
            margin-bottom:12px;
        ">
            <div style="
                display:inline-block;
                padding:7px 16px;
                border-radius:999px;
                background:#f3f7ff;
                color:#1f4e8c;
                font-size:16px;
                font-weight:700;
                border:1px solid #dbe7ff;
            ">
                累计访问人数：{total_visits}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 5. 账户切换辅助函数
# =========================
def is_switch_account_mode() -> bool:
    """
    判断当前是否处于“切换账户”模式。
    """
    try:
        return str(st.query_params.get("switch", "")).strip() == "1"
    except Exception:
        return False


def trigger_switch_account():
    """
    触发切换账户：
    先在 URL 中写入 switch=1，再退出当前账号。
    退出后页面会提示用户登录其他账号。
    """
    st.query_params["switch"] = "1"
    st.logout()


def trigger_normal_logout():
    """
    正常退出登录。
    """
    try:
        if "switch" in st.query_params:
            del st.query_params["switch"]
    except Exception:
        pass
    st.logout()


def render_account_action_buttons(prefix: str):
    """
    渲染“切换账户 / 退出登录”按钮。
    prefix 用于避免不同位置按钮 key 冲突。
    """
    col1, col2 = st.columns(2)

    with col1:
        if st.button("切换账户", use_container_width=True, key=f"{prefix}_switch_account"):
            trigger_switch_account()

    with col2:
        if st.button("退出登录", use_container_width=True, key=f"{prefix}_logout"):
            trigger_normal_logout()


# =========================
# 6. 加载模型、scaler 和训练数据
# =========================
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
# 7. 预测函数
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
# 8. 输入范围检查
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
# 9. DataFrame 导出为 Excel 字节流
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
# 10. 页面样式（尽量保持你原有布局不变）
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
        font-size: 1.3rem;
        color: #444444;
        margin-bottom: 1.2rem;
        text-align: center;
    }

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

    [data-testid="stSidebar"] * {
        font-size: 17px !important;
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
# 11. 页面顶部标题
# =========================
def render_header():
    st.markdown(
        f'<div class="main-title">{CONFIG["app_title_zh"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{CONFIG["subtitle_zh"]}</div>',
        unsafe_allow_html=True,
    )


# =========================
# 12. 未授权时的公开预览说明
# =========================
def render_preview_section():
    mode = get_access_mode()

    st.markdown(
        f'<div class="mode-badge">当前访问模式：{mode}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="preview-card">
        <h3>平台预览</h3>
        <p>本平台支持金属材料应力比R[-1, 1]疲劳强度在线预测。</p>
        <p>当前页面为公开预览区域，所有人都可以查看平台介绍和使用说明。</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="preview-card">
        <h3>使用方式</h3>
        <p><b>public</b>：免费公开版本，所有访问者都能直接使用，目前暂未开通。</p>
        <p><b>authorized</b>：仅授权用户可使用，如需授权，请联系管理员（邮箱：qswang24b@imr.ac.cn）申请授权。</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 13. 未授权时的操作提示
# =========================
def render_access_prompt(user_email: str = ""):
    """
    在 authorized 模式下，如果当前用户还不能使用，
    就显示登录、切换账户和授权说明。
    """
    st.markdown("---")
    st.subheader("登录入口")

    if not auth_is_configured():
        st.warning("当前平台已切换到 authorized 模式，但登录认证尚未配置完成，因此暂时只能预览。")
        st.info("请先在 secrets.toml 中完成 [auth] 配置。")
        return

    # 未登录
    if not safe_is_logged_in():
        if is_switch_account_mode():
            st.info("已退出当前账号，请点击下方按钮登录其他账号。")
        else:
            st.warning("当前为公开预览模式，请先登录；若您的账户已被授权，登录后可直接使用。")

        if st.button("登录并验证身份", use_container_width=True, key="login_entry_button"):
            st.login()
        return

    # 已登录但未授权
    st.warning(f"当前登录账号：{user_email}。你已登录，但尚未获得使用权限。")
    st.info("如需开通权限，请联系管理员将你的邮箱加入 authorized_users。")

    render_account_action_buttons(prefix="unauthorized_user_actions")


# =========================
# 14. 管理员提示区
# =========================
def render_admin_panel():
    st.markdown("---")
    st.subheader("管理员说明")

    st.write("当前平台只保留两种模式：")
    st.write("- public：免费公开")
    st.write("- authorized：通过授权获得使用权")

    st.write("secrets.toml 推荐写法：")
    st.code(
        """
[app_mode]
access_mode = "authorized"

[access_control]
admin_users = ["your_email@example.com"]
authorized_users = ["user1@example.com", "user2@example.com"]
        """.strip()
    )

    st.success("你以后无论是手动开通，还是想主动赠送某个账户免费权限，都只需要把对方邮箱加入 authorized_users。")


# =========================
# 15. 正式预测平台
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
        st.markdown("---")

        st.subheader("💡 说明")
        for note in CONFIG["notes"]:
            st.write(f"- {note}")

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
                "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
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
        st.subheader("训练数据范围")

        range_df = pd.DataFrame({
            "Feature": CONFIG["feature_names"],
            "Chinese": [CONFIG["feature_labels_zh"][f] for f in CONFIG["feature_names"]],
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
# 16. 主程序入口
# =========================
def main():
    """
    主逻辑：
    1) 永远先显示页面标题
    2) 标题下方显示累计访问人数
    3) public 模式 -> 直接开放
    4) authorized 模式 -> 只有授权用户可用
    """
    render_global_style()
    render_header()
    render_visitor_counter()

    mode = get_access_mode()

    if mode == "public":
        st.success("当前模式为 public：所有访问者都可直接使用正式预测功能。")
        render_predictor_app()
        return

    if mode == "authorized":
        if not auth_is_configured():
            render_preview_section()
            render_access_prompt()
            return

        if not safe_is_logged_in():
            render_preview_section()
            render_access_prompt()
            return

        user_email = get_current_user_email()

        if is_authorized_user(user_email):
            st.success(f"当前登录账号：{user_email}；已获得使用权限")

            render_account_action_buttons(prefix="authorized_user_actions")

            admin_users, _ = get_access_lists()
            if user_email in admin_users:
                render_admin_panel()

            render_predictor_app()
            return

        render_preview_section()
        render_access_prompt(user_email=user_email)
        return


# =========================
# 17. 程序入口
# =========================
if __name__ == "__main__":
    main()
