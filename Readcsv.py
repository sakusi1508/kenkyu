import os
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

def preprocess_text_japanese(text):
    mecab = MeCab.Tagger("-Owakati")  # 分かち書きオプション
    tokens = mecab.parse(text).strip().split()
    stopwords = set([
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "も", "な", "だ", "する", "ある", "いる", "こと", "これ", "それ", "あれ", "どれ", "ため", "ここ", "そこ", "あそこ", "もの", "よ", "う"
    ])
    return [word for word in tokens if word not in stopwords]

def score_reviews_csv(input_csv_path, output_csv_path, model_path, threshold=0.5):
    """レビューCSVを読み込み、関連キーワードとの類似度スコアをCSV出力する。

    Parameters
    ----------
    input_csv_path : str
        入力レビューCSVの絶対パス（'review'列が必要）。
    output_csv_path : str
        出力スコアCSVの絶対パス。
    model_path : str
        日本語Word2Vecモデルのパス（.bin）。
    threshold : float
        類似度の閾値。

    Returns
    -------
    str
        保存されたCSVの絶対パス。
    """
    df_reviews = pd.read_csv(input_csv_path)
    if 'review' not in df_reviews.columns:
        raise ValueError("入力CSVに 'review' 列が見つかりません")
    review_texts = df_reviews['review'].dropna().tolist()

    all_words = []
    for text in review_texts:
        all_words.extend(preprocess_text_japanese(text))
    word_counts = Counter(all_words)

    word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

    scores = {}
    for rating, keywords in ratings.items():
        similarities = []
        for word in word_counts.keys():
            if word in word2vec:
                word_vector = word2vec[word].reshape(1, -1)
                keyword_vectors = [word2vec[k].reshape(1, -1) for k in keywords if k in word2vec]
                if keyword_vectors:
                    sim_scores = [cosine_similarity(word_vector, kv)[0][0] for kv in keyword_vectors]
                    max_sim = max(sim_scores)
                    if max_sim >= threshold:
                        similarities.append(max_sim)
        avg_similarity = np.mean(similarities) if similarities else 0
        scores[rating] = avg_similarity

    df_scores = pd.DataFrame(list(scores.items()), columns=['Rating', 'Score'])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_scores.to_csv(output_csv_path, index=False)
    return output_csv_path

if __name__ == '__main__':
    # 簡易CLI実行例（環境変数で指定）
    input_csv_path = os.environ.get('INPUT_CSV')
    output_csv_path = os.environ.get('OUTPUT_CSV', '/tmp/scores.csv')
    model_path = os.environ.get('MODEL_PATH', '/Users/sakumasin/Documents/vscode/zemi/models/word2vec/entity_vector/entity_vector.model.bin')
    if not input_csv_path:
        raise SystemExit('INPUT_CSV を指定してください')
    saved = score_reviews_csv(input_csv_path, output_csv_path, model_path)
    print('Results saved to', saved)