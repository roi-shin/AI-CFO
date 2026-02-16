# GAIS AI-CFO（未来会計シミュレーター）

中小企業の経営者向けに、決算書の数字を入力するだけで「未来の資金繰り」と「黒字ライン（損益分岐点）」を視覚的にシミュレーションできるコックピットアプリです。Gemini 2.5 Flash によるAI-CFO診断機能も搭載しています。

## 🚀 主な機能

- **1画面コックピット**: 入力から結果まで一望できるUI
- **リアルタイム・シミュレーション**:
  - 投資・固定費の増減 (-100万円 〜 +100万円)
  - 原価率の改善・悪化 (-20% 〜 +20%)
  - 売上目標の変化 (-50% 〜 +50%)
- **視覚的な分析**: 6ヶ月先の資金繰り予測グラフと黒字ラインゲージ
- **AI-CFO診断**: Gemini 2.5 Pro/Flash が経営アドバイスを提供

## 📦 インストール

1. リポジトリをクローンします
   ```bash
   git clone https://github.com/Start-GAIS/AI-CFO.git
   cd AI-CFO
   ```

2. 依存パッケージをインストールします
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 APIキー設定

Gemini API を使用するため、APIキーの設定が必要です。

1. [Google AI Studio](https://aistudio.google.com/) で API キーを取得します。
2. `.streamlit/secrets.toml.example` をコピーして `.streamlit/secrets.toml` を作成します。
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
3. `secrets.toml` 内の `GEMINI_API_KEY` に取得したキーを貼り付けます。

   ```toml
   [secrets]
   GEMINI_API_KEY = "あなたのAPIキー"
   ```

   ⚠️ `.streamlit/secrets.toml` はGitに含まれないよう `.gitignore` で設定されていますが、取り扱いには十分ご注意ください。

## ▶️ 実行方法

以下のコマンドでアプリを起動します。

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。
