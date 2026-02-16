# AI-CFO（未来会計シミュレーター）

中小企業の経営者向けに、決算書の数字を入力するだけで「未来の資金繰り」と「黒字ライン（損益分岐点）」を視覚的にシミュレーションできるコックピットアプリです。Gemini 2.5 Flash によるAI-CFO診断機能も搭載しています。

## 🚀 アプリを利用する（ブラウザで完結）

以下のリンクから、インストール不要ですぐに利用できます。

**[https://gais-ai-cfo.streamlit.app/](https://gais-ai-cfo.streamlit.app/)**

---

## 💡 使い方（3ステップ）

### 1. デモデータで試す
アプリ画面左側のサイドバーにある **「デモデータ」** ボタン（建設業・IT業など）を押してください。架空の決算数値が自動入力され、グラフが描画されます。

### 2. 未来をシミュレーション
画面中央にあるスライダーを動かして、経営シナリオをテストします。
- **固定費の増減**: 投資（人や設備）をした場合の収益圧迫度
- **原価率の変動**: 仕入れコストが上がった際の影響
- **売上目標**: 目標達成時の資金繰り予測

### 3. AI-CFOに相談
画面下部の **「診断を実行する」** ボタンを押すと、現在のシミュレーション結果に基づき、Gemini 2.5 Flash が具体的な経営アドバイス（資金繰りリスクや改善点）を提示します。

---

## 🛠 技術スタック
- **Frontend/Backend**: [Streamlit](https://streamlit.io/)
- **AI Model**: [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) via Google AI Studio
- **Visualization**: Plotly

## 💻 ローカルでの実行（開発者向け）
自社の環境で動かしたい場合は、リポジトリをクローンして実行してください。

1. リポジトリをクローン
   ```bash
   git clone https://github.com/Start-GAIS/AI-CFO.git
   cd AI-CFO
   ```
2. 依存パッケージをインストール
   ```bash
   pip install -r requirements.txt
   ```
3. APIキーの設定 (`.streamlit/secrets.toml`)
   ```toml
   [secrets]
   GEMINI_API_KEY = "あなたのAPIキー"
   ```
4. アプリ起動
   ```bash
   streamlit run app.py
   ```
