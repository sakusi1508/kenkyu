import pandas as pd
import numpy as np
from typing import Dict, List, Any

# 5つのゲームタイプ（クラスター C1-C5）ごとの14因子の理想的なパターンを定義
# (前回の修正を反映済み)
EMOTIONAL_PATTERNS = {
    'C1:ファンタジー型': {
        'RPG性': 3, 'シナリオの重要度': 3, '遂行時間の長さ': 3, 'アクション性': 1, 'パズル性': 3,
        '報酬の程度': 1, 'グラフィックのリアル性': 1, '音響効果': 2, '操作のリアクション': 1,
        '世界観の仮想・現実性': 1, 'キャラクターの知名度': 2, '文化の異質・同質感': 1,
        '参加人数の多さ': 2, 'ハードウェア普及度': 2
    },
    'C2:映画型': {
        'RPG性': 3, 'シナリオの重要度': 3, '遂行時間の長さ': 3, 'アクション性': 3, 'パズル性': 1,
        '報酬の程度': 1, 'グラフィックのリアル性': 3, '音響効果': 3, '操作のリアクション': 3,
        '世界観の仮想・現実性': 3, 'キャラクターの知名度': 2, '文化の異質・同質感': 3,
        '参加人数の多さ': 2, 'ハードウェア普及度': 2
    },
    'C3:現実活動型': {
        'RPG性': 1, 'シナリオの重要度': 1, '遂行時間の長さ': 1, 'アクション性': 3, 'パズル性': 1,
        '報酬の程度': 3, 'グラフィックのリアル性': 3, '音響効果': 3, '操作のリアクション': 3,
        '世界観の仮想・現実性': 3, 'キャラクターの知名度': 2, '文化の異質・同質感': 3,
        '参加人数の多さ': 2, 'ハードウェア普及度': 2
    },
    'C4:キャラクター型': {
        'RPG性': 1, 'シナリオの重要度': 1, '遂行時間の長さ': 1, 'アクション性': 2, 'パズル性': 2,
        '報酬の程度': 3, 'グラフィックのリアル性': 1, '音響効果': 2, '操作のリアクション': 2,
        '世界観の仮想・現実性': 1, 'キャラクターの知名度': 3, '文化の異質・同質感': 1,
        '参加人数の多さ': 2, 'ハードウェア普及度': 1
    },
    'C5:シミュレーション型': {
        'RPG性': 2, 'シナリオの重要度': 2, '遂行時間の長さ': 3, 'アクション性': 1, 'パズル性': 3,
        '報酬の程度': 2, 'グラフィックのリアル性': 3, '音響効果': 1, '操作のリアクション': 1,
        '世界観の仮想・現実性': 3, 'キャラクターの知名度': 2, '文化の異質・同質感': 3,
        '参加人数の多さ': 2, 'ハードウェア普及度': 2
    }
}

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """コサイン類似度を計算します。"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def estimate_game_type_percentages(input_csv_path: str, output_csv_path: str):
    """
    CSVファイルを読み込み、ゲームタイトルごとに5つのゲームタイプへの適合率を推定し、CSVに出力します。
    縦長形式 (Rating, Score) のCSVにも対応します。
    """
    try:
        df_input = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_csv_path}' が見つかりません。パスを確認してください。")
        return
    except Exception as e:
        print(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # ★★★ 縦長データの横長データへの変換処理を追加 ★★★
    if all(col in df_input.columns for col in ['Rating', 'Score']):
        print(f"CSVファイルが縦長形式 (Rating, Score) であることを検出しました。横長形式に変換します。")
        
        # RatingをインデックスにしてScoreを値とし、転置して1行のデータフレームにする
        try:
            df_games = df_input.set_index('Rating')['Score'].to_frame().T
        except KeyError as e:
            print(f"致命的なエラー: 変換に必要な列 'Rating' または 'Score' が見つかりませんでした: {e}")
            return
            
        # 1行しかないので、仮の 'game_title' を付与する
        file_name = input_csv_path.split('/')[-1]
        df_games.insert(0, 'game_title', f"Game: {file_name}")
    else:
        # 以前のコードが想定していた横長形式（複数ゲームタイトル）の場合
        df_games = df_input
        print("CSVファイルが横長形式であることを想定し、処理を続行します。")

    # 必須の14因子列が存在するか確認
    factor_list = list(list(EMOTIONAL_PATTERNS.values())[0].keys())
    
    missing_columns = [col for col in factor_list if col not in df_games.columns]
    if missing_columns:
        print(f"致命的なエラー: データに以下の必須の14因子列が不足しています: {', '.join(missing_columns)}")
        print(f"データフレームの列: {list(df_games.columns)}")
        return

    results: List[Dict[str, Any]] = []
    
    # 理想的なパターンの因子ベクトルを一度作成
    pattern_vectors = {
        label: np.array([pattern_dict[key] for key in factor_list]) 
        for label, pattern_dict in EMOTIONAL_PATTERNS.items()
    }

    # 各ゲーム（行）について処理
    for _, row in df_games.iterrows():
        game_title = row['game_title']
        user_vec = row[factor_list].values  # 14因子の値のみを抽出
        
        raw_similarities = {}
        
        # コサイン類似度を計算
        for label, pattern_vec in pattern_vectors.items():
            # 念のためデータ型をfloatに変換してから計算
            similarity = cosine_similarity(user_vec.astype(float), pattern_vec) 
            raw_similarities[label] = similarity
            
        total_similarity = sum(raw_similarities.values())
        result_row = {'game_title': game_title}
        
        # 適合率（パーセンテージ、小数第2位）への変換
        if total_similarity > 0:
            for label, sim in raw_similarities.items():
                percentage = (sim / total_similarity) * 100
                result_row[f'{label}_適合率(%)'] = round(percentage, 2)
        else:
            for label in raw_similarities.keys():
                result_row[f'{label}_適合率(%)'] = 0.00
            
        results.append(result_row)
        
    df_results = pd.DataFrame(results)
    
    # 結果をCSVとして保存
    df_results.to_csv(output_csv_path, index=False)
    
    print(f"\n推定が完了しました。全結果は '{output_csv_path}' に保存されました。")

# -----------------------------------------------------------
## 実行部

# 【利用方法】以下のパスをあなたのCSVファイルのパスに置き換えて、プログラムを実行してください。

# ★★★ ここを実際のCSVファイルパスに変更 ★★★
input_csv_path = '/Users/sakumasin/Documents/vscode/zemi/score/supura_scores_japanese.csv' # 添付されたファイル名
output_csv_path = '/Users/sakumasin/code/kenkyu/14-5/result/supu14-5.csv' # 出力ファイル名

estimate_game_type_percentages(input_csv_path, output_csv_path)