## Webフロー概要

1. ステップ1: SteamアプリIDと出力CSV名を入力し、レビューCSVを生成します（`Getcsv.py`）。
2. ステップ2: 生成したレビューCSVと出力スコアCSV名を入力し、評定スコアCSVを生成します（`Readcsv.py`）。

バックエンドは Flask を利用予定です。
steamAPIをもちいてゲームのレビューデータをスクレイピング『getCSV.py』

14因子それぞれにキーワードを定義してそれとのコサイン類似度を算出する『readCSV』

# キーワード

    'RPG性': ['クエスト', 'キャラクター', 'ストーリー', 'レベル'],
    'アクション性': ['アクション', '戦闘', '速い', '戦う'],
    'パズル性': ['パズル', '論理', '解く', '頭脳'],
    '報酬の程度': ['報酬', '解除', '達成', '進捗'],
    'グラフィックのリアル性': ['グラフィック', 'ビジュアル', 'リアル', '詳細'],
    '音響効果': ['音響', 'オーディオ', '音楽', '声'],
    '世界観の仮想・現実性': ['世界', '環境', '仮想', '現実'],
    'キャラクターの知名度': ['キャラクター', 'ヒーロー', 'NPC', '悪役'],
    '文化の異質・同質感': ['文化', 'ファンタジー', '現実', '設定'],
    '操作のリアクション': ['操作', '反応', '入力', '移動'],
    '遂行時間の長さ': ['長い', '時間', '持続', '期間'],
    'シナリオの重要度': ['シナリオ', 'プロット', '物語', '対話'],
    '参加人数の多さ': ['マルチプレイヤー', 'オンライン', '協力', 'チーム'],
    'ハードウェア普及度': ['PC', 'コンソール', 'プラットフォーム', 'ハードウェア']


# mecabを用いてストップワードを削除
def preprocess_text_japanese(text):
    mecab = MeCab.Tagger("-Owakati")  # 分かち書きオプション
    tokens = mecab.parse(text).strip().split()
    #日本語ストップワード（必要に応じて追加）
    stopwords = set([
    
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "も", "な", "だ", "する", "ある", "いる", "こと", "これ", "それ", "あれ", "どれ", "ため", "ここ", "そこ", "あそこ", "もの", "よ","う"])
    
  tokens = [word for word in tokens if word not in stopwords]
    return tokens


# 因子解析

```python
import requests
import pandas as pd

def get_steam_reviews(app_id, review_type='all', num_reviews=10, language='all', api_key='86B191F42E308C0E8245F62118A53A04'):
    base_url = f'http://store.steampowered.com/appreviews/{app_id}?json=1'
    params = {
        'filter': review_type,
        'num_per_page': num_reviews,
        'language': language,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        reviews = response.json()
        return reviews
    else:
        print(f'レビューの取得に失敗しました: {response.status_code}, {response.text}')

# 使用例
app_id = 570940 # 使用するSteamのアプリID
api_key = '86B191F42E308C0E8245F62118A53A04'  # 自分のSteam APIキー
reviews = get_steam_reviews(app_id, review_type='positive', num_reviews=10, language='japanese', api_key=api_key)

# 修正した部分: DataFrameの作成
df = pd.DataFrame(reviews['reviews'])  # レスポンスの'reviews'キーにレビューのデータが含まれていると仮定

# CSVファイルに保存
df.to_csv('/Users/sakumasin/Documents/vscode/zemi/csv/dakuso_review.csv', index=False)
print('csv完了')

```

```python
import MeCab #日本語のストップワード削除
from gensim.models import KeyedVectors #Word2vec処理
from sklearn.metrics.pairwise import cosine_similarity #コサイン類似度を計算
import numpy as np
from collections import Counter #リスト型処理
import pandas as pd
#日本語の関連キーワード
ratings = {
    'RPG性': ['クエスト', 'キャラクター', 'ストーリー', 'レベル'],
    'アクション性': ['アクション', '戦闘', '速い', '戦う'],
    'パズル性': ['パズル', '論理', '解く', '頭脳'],
    '報酬の程度': ['報酬', '解除', '達成', '進捗'],
    'グラフィックのリアル性': ['グラフィック', 'ビジュアル', 'リアル', '詳細'],
    '音響効果': ['音響', 'オーディオ', '音楽', '声'],
    '世界観の仮想・現実性': ['世界', '環境', '仮想', '現実'],
    'キャラクターの知名度': ['キャラクター', 'ヒーロー', 'NPC', '悪役'],
    '文化の異質・同質感': ['文化', 'ファンタジー', '現実', '設定'],
    '操作のリアクション': ['操作', '反応', '入力', '移動'],
    '遂行時間の長さ': ['長い', '時間', '持続', '期間'],
    'シナリオの重要度': ['シナリオ', 'プロット', '物語', '対話'],
    '参加人数の多さ': ['マルチプレイヤー', 'オンライン', '協力', 'チーム'],
    'ハードウェア普及度': ['PC', 'コンソール', 'プラットフォーム', 'ハードウェア']
}
#MeCabで日本語テキストを前処理
def preprocess_text_japanese(text):
    mecab = MeCab.Tagger("-Owakati")  # 分かち書きオプション
    tokens = mecab.parse(text).strip().split()
    #日本語ストップワード（必要に応じて追加）
    stopwords = set([
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "も", "な", "だ", "する", "ある", "いる", "こと", "これ", "それ", "あれ", "どれ", "ため", "ここ", "そこ", "あそこ", "もの", "よ","う"])
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

#CSVファイルからレビューを読み込み
csv_path = "/Users/sakumasin/Documents/vscode/zemi/csv/supura_review.csv"
df_reviews = pd.read_csv(csv_path)
review_texts = df_reviews['review'].dropna().tolist()  #'review'列からデータを取得
'''
　　∧,,∧ 　　 ∧,,∧　　　∧,,▲
　 (,,・∀・) 　 ﾐ,,・∀・ﾐ 　 (;;・∀・)
～(_ｕ,ｕﾉ　＠ﾐ_ｕ,,ｕﾐ　＠(;;;;ｕｕﾉ
評定したいゲームタイトルのCSVに変更↑
'''
#全レビューを前処理して単語リストを作成
all_words = []
for text in review_texts:
    all_words.extend(preprocess_text_japanese(text))
#全単語をカウント
word_counts = Counter(all_words)
print(f"Total unique words: {len(word_counts)}")
#Word2Vecモデルの読み込み（日本語版）
model_path = "/Users/sakumasin/Documents/vscode/zemi/models/word2vec/entity_vector/entity_vector.model.bin"  # 日本語Word2Vecモデルのパス
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
#評定項目ごとに単語とのコサイン類似度を計算
scores = {}
threshold = 0.5  #類似度の閾値
for rating, keywords in ratings.items():
    similarities = []
    for word in word_counts.keys():
        if word in word2vec:  #単語がWord2Vecモデルに含まれているか確認
            word_vector = word2vec[word].reshape(1, -1)
            keyword_vectors = [word2vec[k].reshape(1, -1) for k in keywords if k in word2vec]
            if keyword_vectors:
                #各キーワードとの類似度を計算
                sim_scores = [cosine_similarity(word_vector, kv)[0][0] for kv in keyword_vectors]
                max_sim = max(sim_scores)  #最大類似度を使用
                if max_sim >= threshold:  #閾値を超えた場合のみ記録
                    similarities.append(max_sim)
    #類似度の平均をスコアとする
    avg_similarity = np.mean(similarities) if similarities else 0
    scores[rating] = avg_similarity
#結果を表示
print("\nSimilarity Scores:")
for rating, score in scores.items():
    print(f"{rating}: {score:.2f}")
#'scores' 辞書をDataFrameに変換
output_path = "valo_scores_japanese.csv"
df_scores = pd.DataFrame(list(scores.items()), columns=['Rating', 'Score'])
#CSVファイルとして保存
df_scores.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
```

MinecraftをNLP主観感情評定と主観評価とのMAEを比較する場合

[minecraft_scores_japanese](https://www.notion.so/1d2d89563f3381e89446cd0d7b72cdb2?pvs=21)

[maikura](https://www.notion.so/1d2d89563f3381b2a514ed3404ea83fe?pvs=21)

上記２つのCSVをマージした以下のファイルを評定

[mine](https://www.notion.so/1d2d89563f3381eb8863d765d5849951?pvs=21)

以下はmine.csvを評定するpythonのコード

```python
# CSVを読み込み用
import pandas as pd
# Mean Absolute Error(MAE)用
from sklearn.metrics import mean_absolute_error
# Root Mean Squared Error(RMSE)用
from sklearn.metrics import mean_squared_error
import numpy as np

# CSV読み込み
data = pd.read_csv('/Users/sakumasin/Documents/vscode/zemi/MAEpy/valo.csv')

# 正規化（Min-Maxスケーリング）
label = (data['NLP'] - data['NLP'].min()) / (data['NLP'].max() - data['NLP'].min())
pred = (data['SJT'] - data['SJT'].min()) / (data['SJT'].max() - data['SJT'].min())

# MAE計算
mae = mean_absolute_error(label, pred)
print('MAE : {:.3f}'.format(mae))  # 小数点以下3桁で表示

# RMSE計算
rmse = np.sqrt(mean_squared_error(label, pred))
print('RMSE : {:.3f}'.format(rmse))

# CSVに出力　
df = pd.DataFrame([{
    'MAE': mae,
    'RMSE': rmse
}])
df.to_csv('/Users/sakumasin/Documents/vscode/zemi/MAEpy/valoMAE.csv', index=False, encoding='UTF-8')

```

# 解説

マージしたCSVはNLP主観感情推定(0-1スケール)と主観評価(1-5スケール)で異なるため，Min-Max法を用いて正規化し，MAEを算出する

## なぜMAEなのか

人間が誤差基準を判断するため，誤差を人間にとって一番判断しやすい絶対値を用いる

またRMSE MSEは平方根を取るため，極端な外れ値の影響を受けやすい

よって本研究で聞くことのできた範囲の評定ではMAEが適切であると判断した

## 因子分析の検討と因子数の決定のプロセス
