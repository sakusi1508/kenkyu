# ゲームレコメンデーションシステム

HMMの出力結果から感情を判定し、対応するゲームタイトルをレコメンドするシステムです。

## 概要

このシステムは、HMM（Hidden Markov Model）による感情推定結果を分析し、ユーザーの感情状態に基づいて適切なゲームタイトルをレコメンドします。

### 感情状態

HMMでは以下の4つの感情状態を識別します：

- **感覚運動的興奮**: アクション性が高く、操作のリアクションが重要なゲーム
- **難解・頭脳型**: パズル性が高く、頭脳を使うゲーム
- **和みと癒し**: キャラクターの知名度が高く、リラックスできるゲーム
- **設定状況の魅力**: グラフィックや音響効果、シナリオが重要なゲーム

## ファイル構成

- `emotion_game_mapping.csv`: 感情とゲームタイトルの対応関係データベース
- `recommendation_system.py`: レコメンデーションシステムのコア機能
- `hmm_recommendation.py`: HMMの出力結果を処理してレコメンドする機能
- `process_hmm_and_recommend.py`: HMMの出力結果CSVからレコメンドするスクリプト
- `integrate_hmm_recommendation.py`: キーログから直接HMMで感情を推定してレコメンドするスクリプト
- `game_recommendation_gui.py`: **GUIアプリケーション（統合版）**

## 使用方法

### 🎨 GUIアプリケーション（推奨）

すべての機能を統合したGUIアプリケーションを使用するのが最も簡単です：

```bash
python DB/game_recommendation_gui.py
```

GUIアプリケーションの機能：
- ✅ HMM出力結果CSVからレコメンデーション
- ✅ キーログCSVから直接感情推定とレコメンデーション
- ✅ データベースの更新
- ✅ パラメータ設定（時間窓サイズ、トップNなど）
- ✅ 結果の表示とCSV出力
- ✅ 直感的な操作画面

#### GUIアプリケーションの使い方

1. **アプリケーションの起動**
   ```bash
   python DB/game_recommendation_gui.py
   ```

2. **HMM出力結果からレコメンド**
   - 「HMM出力結果CSV」ボタンをクリックしてファイルを選択
   - 必要に応じて「感情列名」を設定（デフォルト: `推定感情`）
   - 「レコメンド数 (Top N)」を設定（デフォルト: 5）
   - 「レコメンデーション実行」ボタンをクリック

3. **キーログから直接処理**
   - 「キーログCSV (オプション)」ボタンをクリックしてファイルを選択
   - 「HMMモデルファイル」ボタンをクリックしてモデルファイルを選択
   - 「時間窓サイズ (秒)」を設定（デフォルト: 30.0）
   - 「レコメンデーション実行」ボタンをクリック

4. **データベースの更新**
   - 「データベース更新」ボタンをクリック
   - MAEgumi.csvから自動的にデータベースを更新

5. **結果の確認**
   - 結果は画面下部のテキストエリアに表示されます
   - 必要に応じて「出力CSV」を指定して結果を保存

### 1. データベースの更新

まず、MAEgumi.csvから感情とゲームタイトルの対応関係を更新します：

```bash
python DB/recommendation_system.py
```

または、GUIアプリケーションの「データベース更新」ボタンを使用できます。

### 2. HMMの出力結果からレコメンド

HMMの出力結果CSVファイルからゲームをレコメンド：

```bash
python DB/process_hmm_and_recommend.py --input mizuki2025/来場様用結果/emotion_analysis_outputaikawa.csv
```

または、より詳細なオプションを指定：

```bash
python DB/process_hmm_and_recommend.py \
  --input mizuki2025/来場様用結果/emotion_analysis_outputaikawa.csv \
  --emotion-column 推定感情 \
  --top-n 5 \
  --output result/recommendation_result.csv
```

### 2-1. キーログから直接処理（オプション）

キーログCSVファイルから直接HMMで感情を推定してレコメンド：

```bash
python DB/integrate_hmm_recommendation.py --keylog keydata/keylog_output.csv
```

HMMモデルのパスを指定：

```bash
python DB/integrate_hmm_recommendation.py \
  --keylog keydata/keylog_output.csv \
  --model HMMleran/hmm_emotion_model_4states.pkl \
  --top-n 5 \
  --output result/recommendation_result.csv
```

### 3. 感情ラベルのリストからレコメンド（テスト用）

感情ラベルのリストから直接レコメンド：

```bash
python DB/process_hmm_and_recommend.py --emotions "設定状況の魅力,設定状況の魅力,和みと癒し"
```

## システムの動作

1. **感情の判定**: HMMの出力結果（感情ラベルの時系列）から、最も多く出現した感情を判定します。
   - 「設定状況の魅力」が最も多い場合 → 「設定状況の魅力」と定義されたゲームタイトルをレコメンド
   - 他の感情が最も多い場合 → その感情に対応するゲームタイトルをレコメンド

2. **ゲームのレコメンド**: 判定された感情に対応するゲームタイトルを、類似度スコアの高い順にレコメンドします。

3. **結果の表示**: 以下の情報を表示します：
   - 判定された主要感情
   - 感情の分布（各感情の出現回数とパーセンテージ）
   - レコメンドされるゲームタイトルのリスト

## データベースの構造

`emotion_game_mapping.csv`には以下の列が含まれます：

- `game_title`: ゲームタイトル
- `emotion_label`: 感情ラベル（HMMの4つの感情状態のいずれか）
- `similarity_score`: 類似度スコア（0.0-1.0）

## 注意事項

- HMMの出力結果CSVファイルには、感情ラベルが含まれている列が必要です（デフォルト: `推定感情`）
- データベースには、HMMの4つの感情状態に対応するゲームタイトルが登録されている必要があります
- 「ゲーム本来の楽しさ」に分類されたゲームは、HMMの4つの感情のうち最も類似度が高いものにマッピングされます
- キーログからの直接処理には `hmmlearn` ライブラリが必要です（`pip install hmmlearn`）
- GUIアプリケーションには `tkinter` が必要です（Python標準ライブラリに含まれています）

## 今後の拡張

- より多くのゲームタイトルの追加
- 感情の重み付けによるレコメンデーション
- ユーザーの過去のプレイ履歴を考慮したレコメンデーション
- Webアプリケーションへの統合

