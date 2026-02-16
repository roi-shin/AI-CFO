"""
GAIS AI-CFO（未来会計シミュレーター）
=======================================
決算書の数字を入力し、経営シナリオの感度分析を行うコックピット。
Gemini 2.5 Flash による AI-CFO 診断付き。
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# ─────────────────────────────────────
# ページ設定
# ─────────────────────────────────────
st.set_page_config(
    page_title="GAIS AI-CFO｜未来会計シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────
def get_step_size(val):
    if val >= 100_000_000: return 1_000_000
    if val >= 10_000_000:  return 100_000
    if val >= 1_000_000:   return 10_000
    return 1_000

def jp_format(val):
    abs_val = abs(val)
    if abs_val >= 100_000_000:
        return f"{val/100_000_000:.1f}億円"
    elif abs_val >= 10_000:
        return f"{val/10_000:.0f}万円"
    else:
        return f"{val:,.0f}円"

# Helper: Generate Tooltip HTML
def get_tooltip_html(help_text):
    if not help_text:
        return ""
    return f'''<div class="tooltip-container">
        <svg class="tooltip-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V11H13V13ZM12 13C11.45 13 11 12.55 11 12C11 11.45 11.45 11 12 11C12.55 11 13 11.45 13 12C13 12.55 12.55 13 12 13ZM12 9C10.9 9 10 9.9 10 11H12C12 12.1 12.9 13 14 13C15.1 13 16 12.1 16 11C16 9.9 15.1 9 14 9H12ZM12 5C11.45 5 11 4.55 11 4C11 3.45 11.45 3 12 3C12.55 3 13 3.45 13 4C13 4.55 12.55 5 12 5Z" fill="currentColor" opacity="0" />
            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2ZM13 16C13 16.5523 12.5523 17 12 17C11.4477 17 11 16.5523 11 16C11 15.4477 11.4477 15 12 15C12.5523 15 13 15.4477 13 16ZM12 14C11.4477 14 11 13.5523 11 13V11C11 10.4477 11.4477 10 12 10C12.5523 10 13 10.4477 13 11V13C13 13.5523 12.5523 14 12 14ZM16.0006 12C16.0368 12.0002 16.0732 12.0002 16.1095 12L16 12C16.0002 12 16.0004 12 16.0006 12ZM14.94 6.88C15.3542 7.29421 15.3542 7.96579 14.94 8.38C14.5258 8.79421 13.8542 8.79421 13.44 8.38C13.0495 7.98947 12.4162 7.98947 12.0257 8.38C11.6351 8.77053 11.6351 9.40384 12.0257 9.79437L12.0306 9.79932L12.0366 9.80528L12.9814 10.75C13.4563 11.2249 13.4563 11.9949 12.9814 12.4699C12.5065 12.9448 11.7365 12.9448 11.2616 12.4699L10.3168 11.5251L10.3113 11.5196C9.28014 10.4884 9.28014 8.81628 10.3113 7.78508C11.3425 6.75389 13.0146 6.75389 14.0458 7.78508L14.94 6.88Z" fill="currentColor" fill-opacity="0" />
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM12 13C12 13 12 13 12 13C12 13 12 13 12 13C11 13 11 12.5 11 12C11 10.5 13 10.5 13 9C13 8.5 12.5 8 12 8C11.5 8 11 8.5 11 9H9C9 7.5 10.5 6 12 6C13.5 6 15 7.5 15 9C15 11 13 11 13 12V13Z" fill="#94A3B8"/>
        </svg>
        <span class="tooltip-text">{help_text}</span>
    </div>'''

# Helper function for Custom Input Label
def custom_label(label, help_text=""):
    tooltip_html = get_tooltip_html(help_text)
    return f'''<div class="input-label-row">{label}{tooltip_html}</div>'''

# Helper function for Custom Metric Card
def custom_metric(label, value, sub="", help_text="", color_type="neutral"):
    # color_type: "positive" (Green), "negative" (Red), "neutral" (Dark Blue/Black)
    if color_type == "positive":
        val_class = "metric-value-positive"
    elif color_type == "negative":
        val_class = "metric-value-negative"
    else:
        val_class = "metric-value-neutral"
    
    # Tooltip HTML
    tooltip_html = get_tooltip_html(help_text)

    return f'''
    <div class="custom-metric-card">
        <div class="metric-label-row">
            <span class="metric-label">{label}</span>
            {tooltip_html}
        </div>
        <div class="metric-value {val_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    '''

# ─────────────────────────────────────
# 回帰コールバック（スライダー同期）
# ─────────────────────────────────────
if "invest" not in st.session_state: st.session_state["invest"] = 0
if "sales_change" not in st.session_state: st.session_state["sales_change"] = 0

def update_invest_from_slider():
    st.session_state["invest"] = st.session_state["invest_slider"]
    st.session_state["invest_number"] = st.session_state["invest_slider"]

def update_invest_from_number():
    st.session_state["invest"] = st.session_state["invest_number"]
    st.session_state["invest_slider"] = st.session_state["invest_number"]

def update_sales_from_slider():
    st.session_state["sales_change"] = st.session_state["sales_slider"]
    st.session_state["sales_number"] = st.session_state["sales_slider"]
    
def update_sales_from_number():
    st.session_state["sales_change"] = st.session_state["sales_number"]
    st.session_state["sales_slider"] = st.session_state["sales_number"]


# ─────────────────────────────────────
# カスタム CSS
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif;
    color: #333;
}
.block-container { padding-top: 1rem; max-width: 1200px; }

/* ── ヘッダー ── */
.main-header {
    background: linear-gradient(135deg, #1A365D 0%, #2A4365 100%);
    color: #fff;
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.15);
    text-align: center;
}
.main-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; color: #fff; }
.main-header p  { margin: .4rem 0 0; font-size: 0.88rem; opacity: 0.9; }

/* ── セクション見出し ── */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1A365D;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1.2rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-badge {
    background-color: #1A365D; color: #fff;
    padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.78rem;
}

/* ── Custom KPI Card (HTML) ── */
.custom-metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-label-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
    line-height: 1.2;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-value-positive { color: #10B981; }
.metric-value-negative { color: #EF4444; }
.metric-value-neutral  { color: #1E293B; }

.metric-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Tooltip Styles (Native Look) */
.tooltip-container {
    position: relative;
    display: inline-flex;
    align-items: center;
    cursor: help;
    margin-left: 2px;
}
.tooltip-icon {
    width: 1rem;
    height: 1rem;
    color: #808495;
    transition: color 0.2s;
}
.tooltip-container:hover .tooltip-icon {
    color: #262730;
}
.tooltip-text {
    visibility: hidden;
    width: max-content;
    max-width: 250px;
    background-color: #262730;
    color: #ffffff;
    text-align: left;
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    position: absolute;
    z-index: 100;
    bottom: 150%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.2s ease-in-out;
    font-size: 0.85rem;
    font-weight: normal;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    white-space: normal;
}
.tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #262730 transparent transparent transparent;
}
.tooltip-container:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* Custom Input Label (Step 1) */
.input-label-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 2px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #475569;
}

/* ── 診断エリア ── */
.diagnosis-box {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-left: 5px solid #1A365D; border-radius: 8px;
    padding: 1.8rem; color: #334155; line-height: 1.7; margin-top: 1rem;
}
.diagnosis-box h3 {
    color: #1A365D; margin-top: 1.2rem; font-size: 1.05rem;
    border-bottom: 1px dashed #CBD5E1; padding-bottom: 0.3rem;
}

/* ── 入力ラベル ── */
div[data-testid="stNumberInput"] label { font-weight: 600; color: #475569; }
.stSlider label { font-weight: 600; color: #475569; }

/* ── 警告バー ── */
.alert-danger {
    background-color: #FEF2F2; border: 1px solid #FCA5A5; color: #B91C1C;
    padding: 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 500;
}
.alert-safe {
    background-color: #ECFDF5; border: 1px solid #6EE7B7; color: #047857;
    padding: 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 500;
}

/* ── サイドバー開閉ボタン常時表示＆位置調整 ── */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000000 !important;
    margin-top: 4px; 
    margin-left: 4px;
}
section[data-testid="stSidebar"] button[kind="hex"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* グラフタイトルの調整 */
.graph-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 0.5rem;
    border-left: 4px solid #64748B;
    padding-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>GAIS AI-CFO ｜ 未来会計シミュレーター</h1>
    <p>数字を入力し、スライダーで経営シナリオを変えると、資金繰りとリスクが可視化されます</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# デモデータ定義
# ─────────────────────────────────────
DEMO_DATA = {
    "construction": {
        "label": "建設業",
        "revenue": 8_000_000,
        "cogs": 5_600_000,
        "fixed_cost": 1_800_000,
        "cash": 5_000_000,
        "receivables": 16_000_000,
        "payables": 5_600_000,
    },
    "it_service": {
        "label": "IT・サービス業",
        "revenue": 5_000_000,
        "cogs": 2_000_000,
        "fixed_cost": 2_500_000,
        "cash": 3_000_000,
        "receivables": 7_500_000,
        "payables": 2_000_000,
    },
    "restaurant": {
        "label": "飲食業",
        "revenue": 3_500_000,
        "cogs": 1_225_000,
        "fixed_cost": 1_900_000,
        "cash": 1_500_000,
        "receivables": 350_000,
        "payables": 612_500,
    },
}

# ─────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────
defaults = {
    "revenue": 5_000_000, "cogs": 2_000_000, "fixed_cost": 2_500_000,
    "cash": 3_000_000, "receivables": 7_500_000, "payables": 2_000_000,
    "industry": "IT・サービス業", 
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────
# サイドバー
# ─────────────────────────────────────
with st.sidebar:
    st.header("デモデータ")
    for key, data in DEMO_DATA.items():
        if st.button(data["label"], key=f"demo_{key}", use_container_width=True):
            for field in defaults:
                if field in data: st.session_state[field] = data[field]
            st.session_state["industry"] = data["label"] # 業界も更新
            st.session_state["invest"] = 0 # リセット
            st.session_state["sales_change"] = 0 # リセット
            # 同期変数のリセット
            st.session_state["sales_slider"] = 0
            st.session_state["sales_number"] = 0
            st.session_state["invest_slider"] = 0
            st.session_state["invest_number"] = 0
            st.rerun()
    st.markdown("---")
    st.header("ストレステスト")
    if st.button("売上 -30% を検証", key="stress_test", use_container_width=True):
        st.session_state["sales_change"] = -30
        st.session_state["sales_slider"] = -30 
        st.session_state["sales_number"] = -30
        st.rerun()


# ─────────────────────────────────────
# STEP 1: 現状の数値入力
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">STEP 1</span> 現状の数値入力</div>', unsafe_allow_html=True)

# 業界選択
st.markdown("##### 🏢 貴社の業種 （AI診断の基準になります）")
st.session_state["industry"] = st.selectbox(
    "industry_hidden",
    ["製造業", "建設業", "IT・サービス業", "飲食業", "小売業", "卸売業", "医療・福祉", "その他"],
    index=["製造業", "建設業", "IT・サービス業", "飲食業", "小売業", "卸売業", "医療・福祉", "その他"].index(st.session_state.get("industry", "その他")) if st.session_state.get("industry") in ["製造業", "建設業", "IT・サービス業", "飲食業", "小売業", "卸売業", "医療・福祉", "その他"] else 7,
    label_visibility="collapsed"
)

col_pl, col_bs = st.columns([1, 1], gap="large")

# ステップ値の動的決定
revenue_step = get_step_size(st.session_state["revenue"])
cogs_step    = get_step_size(st.session_state["cogs"])
fixed_step   = get_step_size(st.session_state["fixed_cost"])
bs_step      = get_step_size(st.session_state["cash"])

with col_pl:
    st.markdown("##### 損益計算書 （月次平均）")
    
    st.markdown(custom_label("月間売上高", "直近の月平均売上高を入力してください（万円単位・税抜）。"), unsafe_allow_html=True)
    st.session_state["revenue"] = st.number_input(
        "月間売上高", min_value=0, step=revenue_step,
        value=st.session_state["revenue"], format="%d", label_visibility="collapsed")
    
    st.markdown(custom_label("変動費（仕入・外注・材料）", "売上増減に比例するコスト（材料費、仕入商品原価、外注費など）。"), unsafe_allow_html=True)
    st.session_state["cogs"] = st.number_input(
        "変動費（仕入・外注・材料）", min_value=0, step=cogs_step,
        value=st.session_state["cogs"], format="%d", label_visibility="collapsed")
    
    st.markdown(custom_label("固定費（家賃・給与・その他）", "売上がゼロでも毎月かかるコスト（家賃、人件費、リース料、水道光熱費など）。"), unsafe_allow_html=True)
    st.session_state["fixed_cost"] = st.number_input(
        "固定費（家賃・給与・その他）", min_value=0, step=fixed_step,
        value=st.session_state["fixed_cost"], format="%d", label_visibility="collapsed")
    
    if st.session_state["revenue"] > 0:
        rate = st.session_state["cogs"] / st.session_state["revenue"]
        st.info(f"変動費率: **{rate:.1%}**")

with col_bs:
    st.markdown("##### 貸借対照表（現在の残高）")
    
    st.markdown(custom_label("現預金残高", "現在の現預金残高を入力してください。"), unsafe_allow_html=True)
    st.session_state["cash"] = st.number_input(
        "現預金残高", min_value=0, step=bs_step,
        value=st.session_state["cash"], format="%d", label_visibility="collapsed")
    
    st.markdown(custom_label("売掛金残高", "まだ回収していない売上の合計額。"), unsafe_allow_html=True)
    st.session_state["receivables"] = st.number_input(
        "売掛金残高", min_value=0, step=bs_step,
        value=st.session_state["receivables"], format="%d", label_visibility="collapsed")
    
    st.markdown(custom_label("買掛金残高", "まだ支払っていない仕入・経費の合計額。"), unsafe_allow_html=True)
    st.session_state["payables"] = st.number_input(
        "買掛金残高", min_value=0, step=bs_step,
        value=st.session_state["payables"], format="%d", label_visibility="collapsed")

    site_parts = []
    if st.session_state["revenue"] > 0:
        site_parts.append(f"売掛回収: {st.session_state['receivables']/st.session_state['revenue']:.1f}ヶ月")
    if st.session_state["cogs"] > 0:
        site_parts.append(f"買掛支払: {st.session_state['payables']/st.session_state['cogs']:.1f}ヶ月")
    if site_parts:
        st.info("  \n".join(site_parts))


# ─────────────────────────────────────
# STEP 2: シナリオ設定
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">STEP 2</span> シナリオ設定（感度分析）</div>', unsafe_allow_html=True)
st.markdown('<span style="color:#d32f2f; font-weight:bold; font-size:0.9rem;">⚠️ 売上が急増する際、運転資金の増加によって一時的に資金が減るリスクがあります。</span>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4, gap="medium")
slider_invest_step = max(10_000, fixed_step // 10)

# スライダーと入力欄の同期（Invest & Sales）

with s1:
    st.markdown("**固定費の増減（月額）**")
    st.slider(
        "invest_slider_hidden", # ラベル非表示（Markdownで自作）
        min_value=-5_000_000, max_value=5_000_000, 
        value=st.session_state.get("invest", 0), 
        step=slider_invest_step,
        key="invest_slider", on_change=update_invest_from_slider,
        label_visibility="collapsed"
    )
    st.number_input(
        "金額指定", 
        value=st.session_state.get("invest", 0), 
        step=slider_invest_step,
        key="invest_number", on_change=update_invest_from_number,
        label_visibility="collapsed"
    )
    if st.session_state.get("invest", 0) != 0: 
        st.markdown(f"**変化額: {jp_format(st.session_state['invest'])}**")

with s2:
    st.markdown("**仕入・外注単価の変動**")
    cost_cut = st.slider(
        "label_cost",
        min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
        format="%+.1f%%",
        help="原価率の変化（－：改善、＋：悪化）",
        label_visibility="collapsed"
    )

with s3:
    st.markdown("**売上目標の変化**")
    st.slider(
        "sales_slider_hidden",
        min_value=-50, max_value=50, 
        value=st.session_state.get("sales_change", 0),
        step=1, format="%+d%%",
        key="sales_slider", on_change=update_sales_from_slider,
        label_visibility="collapsed"
    )
    st.number_input(
        "sales_number_hidden",
        min_value=-50, max_value=50, 
        value=st.session_state.get("sales_change", 0),
        step=1, 
        key="sales_number", on_change=update_sales_from_number,
        label_visibility="collapsed"
    )
    
    target_rev_preview = st.session_state["revenue"] * (1 + st.session_state.get("sales_change", 0) / 100)
    st.markdown(f"**目標: {jp_format(target_rev_preview)}**")

with s4:
    st.markdown("**目標達成期間**")
    ramp_months = st.slider(
        "label_ramp",
        min_value=1, max_value=6, value=1, step=1,
        format="%dヶ月",
        help="売上が目標に到達するまでの期間",
        label_visibility="collapsed"
    )


# ─────────────────────────────────────
# 計算ロジック
# ─────────────────────────────────────
rev = st.session_state["revenue"]
cgs = st.session_state["cogs"]
fxd = st.session_state["fixed_cost"]
csh = st.session_state["cash"]
rec = st.session_state["receivables"]
pay = st.session_state["payables"]
ind = st.session_state["industry"]
invest = st.session_state.get("invest", 0)
sales_change = st.session_state.get("sales_change", 0)

v_rate = cgs / rev if rev > 0 else 0.0
m_rec  = rec / rev if rev > 0 else 0.0
m_pay  = pay / cgs if cgs > 0 else 0.0

target_rev    = rev * (1 + sales_change / 100)
sim_v_rate    = v_rate * (1 + cost_cut / 100)
sim_fxd       = fxd + invest

mg_rate = max(1.0 - sim_v_rate, 0.001)
bep_rev  = sim_fxd / mg_rate
bep_diff = target_rev - bep_rev

target_op_profit = target_rev - (target_rev * sim_v_rate) - sim_fxd
safety_margin_ratio = (bep_diff / target_rev * 100) if target_rev > 0 else 0.0
invest_payback_sales = invest / mg_rate if invest > 0 and mg_rate > 0 else 0.0

months_label = [f"{i}ヶ月" for i in range(7)]
cf_line = [csh]

current_act_csh = csh
prev_ar_balance = rec
prev_ap_balance = pay

for i in range(1, 7):
    if ramp_months <= 1:
        month_rev = target_rev
    else:
        progress = min(i / ramp_months, 1.0)
        month_rev = rev + (target_rev - rev) * progress
    
    month_cgs = month_rev * sim_v_rate
    month_op_profit = month_rev - month_cgs - sim_fxd
    
    curr_ar_balance = month_rev * m_rec
    curr_ap_balance = month_cgs * m_pay
    
    delta_ar = curr_ar_balance - prev_ar_balance
    delta_ap = curr_ap_balance - prev_ap_balance
    
    month_cash_flow = month_op_profit - delta_ar + delta_ap
    
    current_act_csh += month_cash_flow
    cf_line.append(current_act_csh)
    
    prev_ar_balance = curr_ar_balance
    prev_ap_balance = curr_ap_balance

min_cash = min(cf_line)
short_month = next((i for i, x in enumerate(cf_line) if x < 0), None)
months_sales_ratio = min_cash / target_rev if target_rev > 0 else 0

# ─────────────────────────────────────
# RESULT: 診断結果
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">RESULT</span> 診断結果</div>', unsafe_allow_html=True)



# KPIカード
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(custom_metric(
        label="月次営業利益（目標時）",
        value=jp_format(target_op_profit),
        sub="",
        help_text=f"目標売上 {jp_format(target_rev)} の時の営業利益",
        color_type="positive" if target_op_profit >= 0 else "negative"
    ), unsafe_allow_html=True)

with k2:
    st.markdown(custom_metric(
        label="損益分岐点売上高",
        value=jp_format(bep_rev),
        sub="",
        help_text=f"売上 {jp_format(bep_rev)} で収支均衡（利益ゼロ）",
        color_type="neutral"
    ), unsafe_allow_html=True)

with k3:
    st.markdown(custom_metric(
        label="安全余裕率",
        value=f"{safety_margin_ratio:.1f}%",
        sub=f"売上{safety_margin_ratio:.1f}%減まで黒字" if safety_margin_ratio > 0 else "赤字水準",
        help_text="現在の売上がどれだけ減っても赤字にならないかの割合",
        color_type="positive" if safety_margin_ratio > 0 else "negative"
    ), unsafe_allow_html=True)

with k4:
    sub_text = f"{invest_payback_sales/10000:.0f}万円の売上が必要" if invest > 0 else ""
    st.markdown(custom_metric(
        label="投資回収に必要な売上",
        value=jp_format(invest_payback_sales),
        sub=sub_text,
        help_text="増えた固定費（投資）を賄うために必要な追加売上高",
        color_type="neutral"
    ), unsafe_allow_html=True)

with k5:
    st.markdown(custom_metric(
        label="最低預金残高（6ヶ月間）",
        value=jp_format(min_cash),
        sub="" if min_cash > 0 else "資金ショート警告",
        help_text="今後6ヶ月で最も預金が減るタイミングの残高",
        color_type="positive" if min_cash > 0 else "negative"
    ), unsafe_allow_html=True)



st.write("")

# グラフ行
g1, g2 = st.columns([3, 2], gap="large")

# 単位調整ロジック（万円/億円）
max_cash = max(max(cf_line), abs(min(cf_line)))
if max_cash >= 100_000_000:
    unit_str = "億円"
    divider = 100_000_000
else:
    unit_str = "万円"
    divider = 10_000

y_cf_scaled = [v / divider for v in cf_line]

with g1:
    st.markdown(f'<div class="graph-header">【推移】資金繰り予測 ({unit_str}単位)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    # 軸の最小値調整（ショート時）
    min_y_scaled = min(min(y_cf_scaled), -100) if min(y_cf_scaled) < 0 else 0
    
    fig.add_hrect(y0=min_y_scaled, y1=0, fillcolor="#FEF2F2", opacity=0.8, layer="below", line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="0", annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=months_label, y=y_cf_scaled, mode='lines+markers',
        line=dict(color='#1A365D', width=3),
        marker=dict(size=8, color=['#EF4444' if x < 0 else '#1A365D' for x in cf_line]),
        name="現預金推移",
        text=[jp_format(v) for v in cf_line], hovertemplate='%{x}<br>残高: %{text}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title=f"現預金残高 ({unit_str})",
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.markdown('<div class="graph-header">【安全性】目標売上と損益分岐点売上高の距離</div>', unsafe_allow_html=True)
    max_range = max(target_rev, bep_rev) * 1.3
    # こちらも単位調整
    max_range_scaled = max_range / divider
    target_rev_scaled = target_rev / divider
    bep_rev_scaled = bep_rev / divider
    
    fig2 = go.Figure()
    fig2.add_shape(type="rect", x0=0, x1=bep_rev_scaled, y0=0, y1=1, xref="x", yref="paper",
                   fillcolor="#FFE4E6", line_width=0, opacity=0.5) 
    fig2.add_shape(type="rect", x0=bep_rev_scaled, x1=max_range_scaled, y0=0, y1=1, xref="x", yref="paper",
                   fillcolor="#D1FAE5", line_width=0, opacity=0.5) 
    
    fig2.add_trace(go.Bar(
        x=[target_rev_scaled], y=["売上"], orientation='h',
        marker_color="#1A365D", width=0.5,
        name="目標売上", text=jp_format(target_rev), textposition='auto'
    ))
    
    fig2.add_vline(x=bep_rev_scaled, line_width=3, line_color="#EF4444", line_dash="dash")
    
    fig2.add_annotation(x=bep_rev_scaled, y=1.05, xref="x", yref="paper",
                        text=f"損益分岐点\n{jp_format(bep_rev)}", showarrow=False, 
                        font=dict(color="#EF4444", size=12), xanchor="left")

    fig2.update_layout(
        xaxis=dict(range=[0, max_range_scaled], visible=False),
        yaxis=dict(visible=False),
        height=250, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────
# AI-CFO 診断
# ─────────────────────────────────────
st.markdown("---")
col_btn, col_res = st.columns([1, 4])

with col_btn:
    st.markdown("### AI-CFO 相談")
    st.write("シミュレーション結果をもとに、AIが経営アドバイスを生成します。")
    ask_ai = st.button("診断を実行する", type="primary", use_container_width=True)

with col_res:
    if ask_ai:
        api_key = None
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif "secrets" in st.secrets and "GEMINI_API_KEY" in st.secrets["secrets"]:
            api_key = st.secrets["secrets"]["GEMINI_API_KEY"]

        if not api_key:
            st.error("APIキーが設定されていません。.streamlit/secrets.toml を確認してください。")
        else:
            prompt = f"""以下の中小企業（業種: {ind}）のシミュレーション結果を分析し、貴社に向けたアドバイスを作成してください。
なお、ユーザーの役職を特定せず、「社長」などの呼びかけは避け、「貴社」という表現を使用してください。

※厳守事項：利益、不足額、回収日数などの数値は絶対にAI自身で計算・推測しないでください。必ず上記【データ】セクションで渡された数値をそのまま引用して解説してください。

※【超重要】カタカナ語（アップセル、リードタイム、アライアンス、コンセンサスなど）は使用厳禁です。
必ず現場の従業員や中学生でも直感的にわかる、泥臭く平易な日本語（例：お金の回り、ついで買い、待ち時間、最悪の事態）に翻訳して話してください。
ただし、日常的に使われる言葉（リスク、コスト、システムなど）は許容しますが、コンサル用語は徹底して排除してください。

※【超重要】全方位の一般的なコストカット提案（あれもこれもやれ）は絶対にやめてください。
渡されたデータ（特に『固定費増(投資)』や『変動費率』）を見て、利益を圧迫している【最大の要因1つ】を特定し、そこだけをピンポイントで厳しく指摘・メスを入れてください。
（例：投資額が重すぎるなら、その投資計画自体の撤回や延期を強く迫ること、原価が高すぎるなら仕入れの見直しのみを迫ること）

### ① 資金繰りリスクの評価
- 資金推移（6ヶ月間で最も現金が減った時の残高: {jp_format(min_cash)}）を分析し、資金ショートのリスクがあれば警告してください。
- 現預金月商倍率（最低時）が{months_sales_ratio:.1f}ヶ月分あることが、どの程度の安全性（または危険性）を示すのか評価してください。
- ショートや減少の原因が「売上急増による運転資金の増加（黒字倒産リスク）」なのか、「赤字垂れ流しによる資金枯渇」なのかを明確に区別して指摘してください。
- 業界（{ind}）の平均的な回収サイクルと比べて、貴社のサイト（入金{m_rec:.1f}ヶ月、出金{m_pay:.1f}ヶ月）が適正かも一言触れてください。

### ② 財務の健康診断と潜在リスク
- 「変動費率（原価の重さ）」や「固定費の重さ」など、なぜそのような利益構造になっているのかという【根本原因】を分析してください。
- 安全余裕率は「{safety_margin_ratio:.1f}%」です。{ind}としてこの数値が安全圏か評価してください。

### ③ 明日からやるべき具体的戦術
- 精神論禁止。最大の要因を解決するための具体的アクションを3つ提示してください。

【データ】
- 業種: {ind}
- 売上: {jp_format(rev)} -> {jp_format(target_rev)} ({sales_change:+d}%)
- 変動費率（原価率）: {sim_v_rate:.1%}
- 固定費増（社長の決断した投資額）: {jp_format(invest)}
- 損益分岐点売上高: {jp_format(bep_rev)}
- 6ヶ月後残高: {jp_format(cf_line[-1])}
- 資金ショート: {"あり（黒字倒産リスク）" if short_month else "なし"}
- 現預金月商倍率（最低時）: {months_sales_ratio:.1f}ヶ月
"""
            with st.spinner("AI-CFOがデータを分析中..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    if response and response.candidates and response.candidates[0].content.parts:
                        st.markdown(f'<div class="diagnosis-box">{response.text}</div>',
                                    unsafe_allow_html=True)
                    else:
                         st.error("AIからの回答が空でした。入力内容を見直すか、しばらく待ってから再試行してください。")
                except Exception as e:
                    st.error(f"AI診断中にエラーが発生しました: {e}")
