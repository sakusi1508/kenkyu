import pandas as pd
import numpy as np

# 論文の知見に基づき、5つの感情ラベルごとの14因子の理想的なパターンを定義
# 参考文献: 「コンピュータゲームの特性と楽しさの分析」（山下利之ほか、2004）
emotional_patterns = {
    '設定状況の魅力': {
        'RPG性': 4,
        'アクション性': 2,
        'パズル性': 2,
        '報酬の程度': 3,
        'グラフィックのリアル性': 5,
        '音響効果': 5,
        '世界観の仮想・現実性': 4,
        'キャラクターの知名度': 4,
        '文化の異質・同質感': 3,
        '操作のリアクション': 3,
        '遂行時間の長さ': 4,
        'シナリオの重要度': 5,
        '参加人数の多さ': 2,
        'ハードウェア普及度': 4
    },
    'ゲーム本来の楽しさ': {
        'RPG性': 3,
        'アクション性': 4,
        'パズル性': 3,
        '報酬の程度': 5,
        'グラフィックのリアル性': 3,
        '音響効果': 3,
        '世界観の仮想・現実性': 3,
        'キャラクターの知名度': 4,
        '文化の異質・同質感': 3,
        '操作のリアクション': 4,
        '遂行時間の長さ': 3,
        'シナリオの重要度': 3,
        '参加人数の多さ': 3,
        'ハードウェア普及度': 3
    },
    '感覚運動的興奮': {
        'RPG性': 1,
        'アクション性': 5,
        'パズル性': 1,
        '報酬の程度': 4,
        'グラフィックのリアル性': 4,
        '音響効果': 4,
        '世界観の仮想・現実性': 2,
        'キャラクターの知名度': 3,
        '文化の異質・同質感': 3,
        '操作のリアクション': 5,
        '遂行時間の長さ': 2,
        'シナリオの重要度': 2,
        '参加人数の多さ': 4,
        'ハードウェア普及度': 4
    },
    '和みと癒し': {
        'RPG性': 4,
        'アクション性': 2,
        'パズル性': 2,
        '報酬の程度': 3,
        'グラフィックのリアル性': 3,
        '音響効果': 3,
        '世界観の仮想・現実性': 3,
        'キャラクターの知名度': 5,
        '文化の異質・同質感': 3,
        '操作のリアクション': 2,
        '遂行時間の長さ': 4,
        'シナリオの重要度': 4,
        '参加人数の多さ': 3,
        'ハードウェア普及度': 4
    },
    '難解・頭脳型': {
        'RPG性': 4,
        'アクション性': 1,
        'パズル性': 5,
        '報酬の程度': 2,
        'グラフィックのリアル性': 3,
        '音響効果': 3,
        '世界観の仮想・現実性': 4,
        'キャラクターの知名度': 2,
        '文化の異質・同質感': 2,
        '操作のリアクション': 1,
        '遂行時間の長さ': 5,
        'シナリオの重要度': 4,
        '参加人数の多さ': 1,
        'ハードウェア普及度': 4
    }
}

# コサイン類似度を計算する関数
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

# CSVファイルを読み込み、感情ラベルを推定する関数
def estimate_emotional_labels(input_csv_path, output_csv_path):
    try:
        df_games = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"エラー: 指定されたファイル '{input_csv_path}' が見つかりません。")
        return
    
    results = []
    
    for index, row in df_games.iterrows():
        game_title = row['game_title']
        user_vec = row.drop('game_title').values
        
        similarities = {}
        for label, pattern_dict in emotional_patterns.items():
            pattern_vec = np.array([pattern_dict.get(key, 0) for key in row.drop('game_title').keys()])
            similarity = cosine_similarity(user_vec, pattern_vec)
            similarities[label] = round(similarity, 2)
            
        most_similar_label = max(similarities, key=similarities.get)
        
        result_row = {
            'game_title': game_title,
            'estimated_emotional_label': most_similar_label
        }
        result_row.update({f'similarity_{label}': sim for label, sim in similarities.items()})
        results.append(result_row)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    
    print(f"推定が完了しました。結果は '{output_csv_path}' に保存されました。")

# 使用例
input_csv_path = '/Users/sakumasin/code/kenkyu/syuyougametitle_score/merged_game_scores.csv'  
output_csv_path = '/Users/sakumasin/code/kenkyu/gametitleSENTEI/5rabel/MAEgumi.csv'

# プログラムを実行
estimate_emotional_labels(input_csv_path, output_csv_path)