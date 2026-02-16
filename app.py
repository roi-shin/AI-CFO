"""
GAIS AI-CFO（未来会計シミュレーター）
=======================================
決算書の数字を入れるだけで、AI導入による「未来の資金繰り」と
「損益分岐点」を視覚的にシミュレーションできる経営コックピット。
Gemini 2.5 Pro による AI-CFO 診断付き。
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
    initial_sidebar_state="expanded", # サイドバーはデフォルトで開く
)

# ─────────────────────────────────────
# カスタム CSS
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700;900&display=swap');

/* ── 全体 ── */
html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif;
    color: #333; /* 文字色は濃いグレーで見やすく */
}
.block-container { padding-top: 1rem; max-width: 1200px; }

/* ── ヘッダー ── */
.main-header {
    background: linear-gradient(135deg, #1A365D 0%, #2A4365 100%); /* ネイビーベース */
    color: #fff;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.15);
    text-align: center;
}
.main-header h1 {
    margin: 0; font-size: 1.8rem; font-weight: 700;
    color: #fff;
}
.main-header p { margin: .5rem 0 0; font-size: 0.9rem; opacity: 0.9; }

/* ── セクション見出し ── */
.section-title {
    font-size: 1.2rem; font-weight: 700; color: #1A365D; /* ネイビー */
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 0.5rem;
    margin: 2.5rem 0 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-badge {
    background-color: #1A365D; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; vertical-align: middle;
}

/* ── KPI カード ── */
.kpi-card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
    height: 100%;
}
.kpi-card .label { font-size: 0.85rem; color: #64748B; margin-bottom: 0.5rem; font-weight: 500;}
.kpi-card .value { font-size: 1.8rem; font-weight: 700; color: #1E293B; }
.kpi-positive { color: #10B981 !important; } /* Green */
.kpi-negative { color: #EF4444 !important; } /* Red */

/* ── 診断エリア ── */
.diagnosis-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #1A365D;
    border-radius: 8px;
    padding: 2rem;
    color: #334155;
    line-height: 1.7;
    margin-top: 1rem;
}
.diagnosis-box h3 { color: #1A365D; margin-top: 1.5rem; font-size: 1.1rem; border-bottom: 1px dashed #CBD5E1; padding-bottom: 0.3rem;}

/* ── 入力フィールドラベル ── */
div[data-testid="stNumberInput"] label {
    font-weight: 600; color: #475569;
}
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

/* ── サイドバー開閉ボタンの常時表示 ── */
/* Streamlitのバージョンによってはセレクタが異なる場合がありますが、代表的なものをカバー */
[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    color: #1A365D !important;
}
section[data-testid="stSidebar"] button[kind="hex"] {
    /* 閉じるボタン（<） */
    display: block !important;
    opacity: 1 !important;
    color: #1A365D !important;
}
/* 開くボタン（>） */
button[kind="header"] {
    display: block !important;
    opacity: 1 !important;
    color: #1A365D !important;
}
div[data-testid="collapsedControl"] {
    display: block !important;
    color: #1A365D !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 GAIS AI-CFO ｜ 未来会計シミュレーター</h1>
    <p>1画面コックピットで経営の未来を予測・診断</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# デモデータ定義
# ─────────────────────────────────────
DEMO_DATA = {
    "construction": {
        "label": "🏢 建設業",
        "revenue": 8_000_000,
        "cogs": 5_600_000,
        "fixed_cost": 1_800_000,
        "cash": 5_000_000,
        "receivables": 16_000_000, # 2ヶ月
        "payables": 5_600_000,
    },
    "it_service": {
        "label": "💻 IT・サービス業",
        "revenue": 5_000_000,
        "cogs": 2_000_000, # 変動費率40%
        "fixed_cost": 2_500_000,
        "cash": 3_000_000,
        "receivables": 7_500_000, # 1.5ヶ月
        "payables": 2_000_000,
    },
    "restaurant": {
        "label": "🍽️ 飲食業",
        "revenue": 3_500_000,
        "cogs": 1_225_000, # 変動費率35%
        "fixed_cost": 1_900_000,
        "cash": 1_500_000,
        "receivables": 350_000, # 現金商売に近い
        "payables": 612_500,
    },
}

# ─────────────────────────────────────
# セッション初期化 (デフォルトはITサービス)
# ─────────────────────────────────────
defaults = {
    "revenue": 5_000_000,
    "cogs": 2_000_000,
    "fixed_cost": 2_500_000,
    "cash": 3_000_000,
    "receivables": 7_500_000,
    "payables": 2_000_000,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────
# サイドバー（デモデータのみ）
# ─────────────────────────────────────
with st.sidebar:
    st.header("🛠️ デモデータ読込")
    st.caption("業種ごとのサンプルを一括ロードします")
    
    # 縦並びボタン
    for key, data in DEMO_DATA.items():
        if st.button(data["label"], key=f"demo_{key}", use_container_width=True):
             for field in ["revenue", "cogs", "fixed_cost", "cash", "receivables", "payables"]:
                st.session_state[field] = data[field]
             st.rerun()
    
    st.info("💡 **使い方**\n\nここでの入力は初期値です。デモを選んだら、右側のパネルで数値を調整してください。")
    st.markdown("---")
    st.caption("👈 左上の「<」でメニューを閉じ、「>」で再度開けます。")


# ─────────────────────────────────────
# メイン画面：STEP 1 現状の把握
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">STEP 1</span> 現状の数値入力（PL/BS）</div>', unsafe_allow_html=True)
st.info("💡 ここで入力された数字は「ひと月あたり（月次平均）」の金額です。")

# 入力フォームを2列×3行ではなく、意味のあるグループで横展開
col_pl, col_bs = st.columns([1, 1], gap="large")

with col_pl:
    st.markdown("##### 📄 損益(PL)情報 <small style='color:#666'>（月次平均）</small>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["revenue"] = st.number_input("ひと月の売上高", min_value=0, step=100000, value=st.session_state["revenue"], format="%d")
    with c2:
        st.session_state["cogs"] = st.number_input("ひと月の売上原価", min_value=0, step=100000, value=st.session_state["cogs"], format="%d")
    with c3:
        st.session_state["fixed_cost"] = st.number_input("ひと月の固定費", min_value=0, step=100000, value=st.session_state["fixed_cost"], format="%d")
    
    # 自動計算の変動費率を表示（入力補助）
    if st.session_state["revenue"] > 0:
        rate = st.session_state["cogs"] / st.session_state["revenue"]
        st.caption(f"📊 変動費率: **{rate:.1%}** （売上に占める原価の割合）")

with col_bs:
    st.markdown("##### 💰 貸借(BS)情報 <small style='color:#666'>（現在の残高）</small>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        st.session_state["cash"] = st.number_input("現預金残高", min_value=0, step=100000, value=st.session_state["cash"], format="%d")
    with c5:
        st.session_state["receivables"] = st.number_input("売掛金残高", min_value=0, step=100000, value=st.session_state["receivables"], format="%d")
    with c6:
        st.session_state["payables"] = st.number_input("買掛金残高", min_value=0, step=100000, value=st.session_state["payables"], format="%d")

    # 自動計算のサイトを表示
    site_msg = []
    if st.session_state["revenue"] > 0:
        m_rec = st.session_state["receivables"] / st.session_state["revenue"]
        site_msg.append(f"売掛回収までの期間: <b>{m_rec:.1f}ヶ月</b>")
    if st.session_state["cogs"] > 0:
        m_pay = st.session_state["payables"] / st.session_state["cogs"]
        site_msg.append(f"買掛支払までの期間: <b>{m_pay:.1f}ヶ月</b>")
    if site_msg:
        st.markdown(f"<small>⏳ {' / '.join(site_msg)}</small>", unsafe_allow_html=True)


# ─────────────────────────────────────
# メイン画面：STEP 2 未来シミュレーション
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">STEP 2</span> 未来シミュレーション（スライダー操作）</div>', unsafe_allow_html=True)
st.info("💡 スライダーを動かすと、**来月以降ずっと** その状態が続くと仮定して計算します。")

# 3つのスライダーを横並びに配置
slider_cols = st.columns(3, gap="medium")

with slider_cols[0]:
    invest = st.slider(
        "🚀 投資、固定費の増減（月額）", 
        min_value=-1000000, max_value=1000000, value=0, step=10000, # 減らす方にも振れるように
        format="¥%d", 
        help="来月以降、固定費を増やしますか？減らしますか？（プラス＝投資増、マイナス＝コストカット）"
    )

with slider_cols[1]:
    cost_cut = st.slider(
        "⚙️ 原価の削減・悪化率（ずっと）", 
        min_value=-20.0, max_value=20.0, value=0.0, step=0.5, # 増える方にも振れるように
        format="%.1f%%",
        help="原価率が何％変化しますか？（マイナス＝改善・削減、プラス＝悪化・値上げ）"
    )

with slider_cols[2]:
    sales_change = st.slider(
        "📈 売上目標の変化（ずっと）", 
        min_value=-50, max_value=50, value=0, step=1, 
        format="%+d%%",
        help="現在の売上に対して、来月以降、毎月何％アップ（ダウン）を目指しますか？"
    )


# ─────────────────────────────────────
# 計算ロジック
# ─────────────────────────────────────
# 入力値の取得（session_stateから）
rev = st.session_state["revenue"]
cgs = st.session_state["cogs"]
fxd = st.session_state["fixed_cost"]
csh = st.session_state["cash"]
rec = st.session_state["receivables"]
pay = st.session_state["payables"]

# 基本係数
v_rate = cgs / rev if rev > 0 else 0.0
# 回転期間（月数）
m_rec = rec / rev if rev > 0 else 0.0
m_pay = pay / cgs if cgs > 0 else 0.0

# シミュレーション計算
sim_rev = rev * (1 + sales_change / 100)
# cost_cut は「削減率」なので、マイナスほど良い（原価が下がる）。
# 逆にプラス（悪化）の場合は原価率が上がる。
# slider label: "原価の削減・悪化率" -> -20% (削減) ... +20% (悪化)
# sim_v_rate = v_rate * (1 + rate) -> if -20%, (1 - 0.2) = 0.8倍になる。正しい。
sim_v_rate = v_rate * (1 + cost_cut / 100) 

sim_cgs = sim_rev * sim_v_rate
sim_fxd = fxd + invest
sim_op_profit = sim_rev - sim_cgs - sim_fxd # 月次営業利益

# 損益分岐点 (BEP)
mg_rate = 1.0 - sim_v_rate # 限界利益率
if mg_rate <= 0: mg_rate = 0.001 # ゼロ除算回避
bep_rev = sim_fxd / mg_rate
bep_diff = sim_rev - bep_rev

# キャッシュフロー予測 (簡易シミュレーション)
months_label = [f"{i}ヶ月後" for i in range(7)]
cf_line = [csh] # 0ヶ月目=現在

current_c = csh
for i in range(1, 7):
    # ベースのキャッシュフロー（営業利益ベース）
    base_flow = sim_op_profit 
    
    # 運転資金（売掛・買掛）の影響によるキャッシュ増減
    # シミュレーション初期（特に回収サイト期間内）は、
    # 過去の売上（変更前）の入金と、新しい売上（変更後）の入金が混在する
    
    # ここでは簡易的に、「回収サイト期間内(m_recヶ月)」は
    # 売上増分が現金化されない（＝利益はあるがキャッシュは増えない）として調整する
    if i <= max(m_rec, 1): 
        gap_rev = sim_rev - rev # 売上増分
        # 増えた売上のうち、まだ現金になっていない分をマイナス
        base_flow -= gap_rev
        
    current_c += base_flow
    cf_line.append(current_c)

# KPI計算
min_cash = min(cf_line)
short_month = next((i for i, x in enumerate(cf_line) if x < 0), None)

# ─────────────────────────────────────
# メイン画面：結果表示
# ─────────────────────────────────────
st.markdown('<div class="section-title"><span class="section-badge">RESULT</span> 診断結果</div>', unsafe_allow_html=True)

# KPI・グラフ・AI診断を配置
# 上段：KPIカード
k1, k2, k3, k4 = st.columns(4)
with k1:
    s_cls = "kpi-positive" if sim_op_profit >= 0 else "kpi-negative"
    st.markdown(f'<div class="kpi-card"><div class="label">月ごとの営業利益(予測)</div><div class="value {s_cls}">¥{sim_op_profit:,.0f}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="label">黒字になる最低売上(月)</div><div class="value">¥{bep_rev:,.0f}</div></div>', unsafe_allow_html=True)
with k3:
    b_cls = "kpi-positive" if bep_diff >= 0 else "kpi-negative"
    sign = "+" if bep_diff >= 0 else ""
    # BEPとの差額 → 安全余裕額
    st.markdown(f'<div class="kpi-card"><div class="label">黒字ラインまでの余裕</div><div class="value {b_cls}">{sign}¥{bep_diff:,.0f}</div></div>', unsafe_allow_html=True)
with k4:
    c_cls = "kpi-positive" if cf_line[-1] >= 0 else "kpi-negative"
    st.markdown(f'<div class="kpi-card"><div class="label">6ヶ月後の現預金残高</div><div class="value {c_cls}">¥{cf_line[-1]:,.0f}</div></div>', unsafe_allow_html=True)

st.write("") # Spacer

# 中段：グラフとゲージ
g_col1, g_col2 = st.columns([3, 2], gap="large")

with g_col1:
    # CFチャート
    fig = go.Figure()
    # 警告ゾーン
    fig.add_hrect(y0=min(min_cash, -1000000), y1=0, fillcolor="#FEF2F2", opacity=0.8, layer="below", line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="資金ショート (0円)", annotation_position="bottom right")
    
    # 折れ線
    fig.add_trace(go.Scatter(
        x=months_label, y=cf_line, mode='lines+markers',
        line=dict(color='#1A365D', width=3),
        marker=dict(size=8, color=['#EF4444' if x < 0 else '#1A365D' for x in cf_line]),
        name="現預金推移"
    ))
    
    fig.update_layout(
        title="<b>資金繰り予測（向こう6ヶ月）</b>",
        xaxis_title="", yaxis_title="現預金残高 (円)",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if short_month:
        st.markdown(f'<div class="alert-danger">⚠️ <b>資金ショート警告</b>: {short_month}ヶ月目に残高がマイナスになります。</div>', unsafe_allow_html=True)

with g_col2:
    # BEPゲージ
    max_val = max(sim_rev, bep_rev) * 1.3
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sim_rev,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': bep_rev, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
        title = {'text': "<b>売上 vs 黒字ライン</b>", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#1A365D"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, bep_rev], 'color': "#FEF2F2"},
                {'range': [bep_rev, max_val], 'color': "#ECFDF5"}],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 4},
                'thickness': 0.75,
                'value': bep_rev}
        }
    ))
    fig2.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────
# 下段：AI-CFO 診断ボタンと結果
# ─────────────────────────────────────
st.write("---")
col_ai_btn, col_ai_res = st.columns([1, 4])

with col_ai_btn:
    st.markdown("### 🤖 AI相談")
    st.write("今のシミュレーション結果について、AI-CFOの意見を聞いてみましょう。")
    ask_ai = st.button("💡 診断を実行する", type="primary", use_container_width=True)

with col_ai_res:
    if ask_ai:
        api_key = None
        if "GEMINI_API_KEY" in st.secrets:
             api_key = st.secrets["GEMINI_API_KEY"]
        elif "secrets" in st.secrets and "GEMINI_API_KEY" in st.secrets["secrets"]:
             api_key = st.secrets["secrets"]["GEMINI_API_KEY"]

        if not api_key:
            st.error("APIキーが設定されていません。.streamlit/secrets.toml を確認してください。")
        else:
            prompt = f"""
あなたはプロのCFOです。以下の中小企業のシミュレーション結果を見て、経営アドバイスをください。
出力はMarkdownで見やすく、以下の3点に絞ってください。

1. **財務の健康診断**: 利益構造や黒字ラインの観点から
2. **キャッシュリスク**: 資金ショートの危険性と対策
3. **戦略アクション**: 経営者が明日からやるべき3つのこと（専門用語禁止、中学生でもわかる言葉で）

【データ】
- 月間売上: {sim_rev:,.0f}円 (目標対比 {sales_change:+d}%)
- 営業利益: {sim_op_profit:,.0f}円 (黒字ラインまで あと{bep_diff:,.0f}円)
- 現預金残高(6ヶ月後): {cf_line[-1]:,.0f}円
- 資金ショート発生月: {"なし" if not short_month else f"{short_month}ヶ月目"}
- 損益分岐点比率: {sim_rev/bep_rev*100:.1f}% (100%超なら黒字)
- 固定費増減(投資): 月額 {invest:+,.0f}円
- 原価率変動: {cost_cut:+.1f}%
            """
            
            with st.spinner("🧠 AI-CFOがデータを分析中..."):
                try:
                    genai.configure(api_key=api_key)
                    # ユーザー指定により gemini-2.5-flash を利用
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    
                    st.markdown(f'<div class="diagnosis-box"><h3>🎓 AI-CFOからの回答</h3>{response.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI診断中にエラーが発生しました: {e}")
