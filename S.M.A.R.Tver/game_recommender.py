"""
5個の型から最も適合するゲームタイトルをレコメンドするモジュール
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# ゲームタイトルと型のマッピング（merged_game_scores.csvから読み込む）
def load_game_scores(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    ゲームスコアCSVを読み込む
    
    Args:
        csv_path: CSVファイルのパス（Noneの場合はデフォルトパスを使用）
    
    Returns:
        ゲームスコアのDataFrame
    """
    if csv_path is None:
        # デフォルトパス
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'syuyougametitle_score',
            'merged_game_scores.csv'
        )
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ゲームスコアファイルが見つかりません: {csv_path}")
    
    return pd.read_csv(csv_path)


def calculate_game_type_similarity(game_scores: pd.DataFrame, game_type_percentages: Dict[str, float]) -> pd.DataFrame:
    """
    各ゲームタイトルと5個の型の類似度を計算
    
    Args:
        game_scores: ゲームスコアのDataFrame（14因子を含む）
        game_type_percentages: 5個の型への適合率
    
    Returns:
        ゲームタイトルと型の類似度を含むDataFrame
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from emotion_converter import GAME_TYPE_PATTERNS, FACTOR_LIST, cosine_similarity
    
    results = []
    
    for _, row in game_scores.iterrows():
        game_title = row['game_title']
        game_factors = {factor: row.get(factor, 0.0) for factor in FACTOR_LIST}
        game_vec = np.array([game_factors.get(factor, 0.0) for factor in FACTOR_LIST])
        
        # 各型との類似度を計算
        type_similarities = {}
        for game_type, pattern_dict in GAME_TYPE_PATTERNS.items():
            pattern_vec = np.array([pattern_dict.get(factor, 0) for factor in FACTOR_LIST])
            similarity = cosine_similarity(game_vec.astype(float), pattern_vec.astype(float))
            type_similarities[game_type] = similarity
        
        # ユーザーの型適合率とゲームの型類似度の重み付きスコアを計算
        weighted_scores = {}
        for game_type in GAME_TYPE_PATTERNS.keys():
            user_type_score = game_type_percentages.get(game_type, 0.0) / 100.0
            game_type_similarity = type_similarities.get(game_type, 0.0)
            # 重み付きスコア = ユーザーの型適合率 × ゲームの型類似度
            weighted_scores[game_type] = user_type_score * game_type_similarity
        
        total_score = sum(weighted_scores.values())
        
        results.append({
            'game_title': game_title,
            'total_score': total_score,
            **{f'{gt}_score': score for gt, score in weighted_scores.items()}
        })
    
    return pd.DataFrame(results)


def recommend_games(
    game_type_percentages: Dict[str, float],
    common_impression_scores: Dict[str, float],
    game_scores_path: Optional[str] = None,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    5個の型から最も適合するゲームタイトルをレコメンド
    
    Args:
        game_type_percentages: 5個の型への適合率
        common_impression_scores: 共通印象スコア
        game_scores_path: ゲームスコアCSVファイルのパス
        top_n: レコメンドするゲームの数
    
    Returns:
        (ゲームタイトル, スコア)のタプルのリスト
    """
    # ゲームスコアを読み込み
    game_scores = load_game_scores(game_scores_path)
    
    # 各ゲームタイトルと型の類似度を計算
    similarity_df = calculate_game_type_similarity(game_scores, game_type_percentages)
    
    # 共通印象スコアを考慮した最終スコアを計算
    final_scores = []
    for _, row in similarity_df.iterrows():
        game_title = row['game_title']
        base_score = row['total_score']
        
        # 共通印象スコアを考慮（最も適合する型の共通印象スコアを加算）
        max_common_impression = max(common_impression_scores.values()) if common_impression_scores else 0.0
        
        # 最終スコア = ベーススコア + 共通印象スコアの重み付き平均
        final_score = base_score * 0.7 + max_common_impression * 0.3
        
        final_scores.append((game_title, final_score))
    
    # スコアでソート
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Top Nを返す
    return final_scores[:top_n]

