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
csv_path = "/Users/sakumasin/Documents/vscode/zemi/csv/OV2__3_review.csv"
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
output_path = "OV2_scores_japanese.csv"
df_scores = pd.DataFrame(list(scores.items()), columns=['Rating', 'Score'])
#CSVファイルとして保存
df_scores.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")