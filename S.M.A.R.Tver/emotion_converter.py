"""
14個の感情ラベル（14因子）を5個の主観感情と5個の型に変換するモジュール
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# 5つの主観感情ごとの14因子の理想的なパターン
EMOTIONAL_PATTERNS = {
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

# 5つのゲームタイプ（クラスター C1-C5）ごとの14因子の理想的なパターン
GAME_TYPE_PATTERNS = {
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

# 14因子のリスト
FACTOR_LIST = [
    'RPG性', 'アクション性', 'パズル性', '報酬の程度', 'グラフィックのリアル性',
    '音響効果', '世界観の仮想・現実性', 'キャラクターの知名度', '文化の異質・同質感',
    '操作のリアクション', '遂行時間の長さ', 'シナリオの重要度', '参加人数の多さ', 'ハードウェア普及度'
]

# 5つの主観感情のリスト
EMOTIONAL_LABELS = list(EMOTIONAL_PATTERNS.keys())

# 5つのゲームタイプのリスト
GAME_TYPES = list(GAME_TYPE_PATTERNS.keys())


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """コサイン類似度を計算します。"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def convert_to_emotional_labels(factor_values: Dict[str, float]) -> Dict[str, float]:
    """
    14因子の値を5個の主観感情への適合度に変換
    
    Args:
        factor_values: 14因子の値を持つ辞書
    
    Returns:
        5個の主観感情への適合度（類似度）を持つ辞書
    """
    # 14因子のベクトルを作成
    user_vec = np.array([factor_values.get(factor, 0.0) for factor in FACTOR_LIST])
    
    similarities = {}
    for label, pattern_dict in EMOTIONAL_PATTERNS.items():
        pattern_vec = np.array([pattern_dict.get(factor, 0) for factor in FACTOR_LIST])
        similarity = cosine_similarity(user_vec.astype(float), pattern_vec.astype(float))
        similarities[label] = similarity
    
    return similarities


def convert_to_game_types(factor_values: Dict[str, float]) -> Dict[str, float]:
    """
    14因子の値を5個の型（C1-C5）への適合率に変換
    
    Args:
        factor_values: 14因子の値を持つ辞書
    
    Returns:
        5個の型への適合率（パーセンテージ）を持つ辞書
    """
    # 14因子のベクトルを作成
    user_vec = np.array([factor_values.get(factor, 0.0) for factor in FACTOR_LIST])
    
    raw_similarities = {}
    for label, pattern_dict in GAME_TYPE_PATTERNS.items():
        pattern_vec = np.array([pattern_dict.get(factor, 0) for factor in FACTOR_LIST])
        similarity = cosine_similarity(user_vec.astype(float), pattern_vec.astype(float))
        raw_similarities[label] = similarity
    
    # 適合率（パーセンテージ）に変換
    total_similarity = sum(raw_similarities.values())
    percentages = {}
    
    if total_similarity > 0:
        for label, sim in raw_similarities.items():
            percentages[label] = (sim / total_similarity) * 100
    else:
        for label in raw_similarities.keys():
            percentages[label] = 0.0
    
    return percentages


def compare_emotional_and_type_similarity(
    emotional_similarities: Dict[str, float],
    game_type_percentages: Dict[str, float]
) -> Dict[str, float]:
    """
    5個の主観感情と5個の型の共通印象を比較
    
    Args:
        emotional_similarities: 5個の主観感情への類似度
        game_type_percentages: 5個の型への適合率
    
    Returns:
        各型と主観感情の組み合わせの共通印象スコア
    """
    # 主観感情と型の対応関係を定義（共通印象のマッピング）
    # ここでは、各型がどの主観感情と関連が深いかを定義
    type_emotion_mapping = {
        'C1:ファンタジー型': ['和みと癒し', '難解・頭脳型', '設定状況の魅力'],
        'C2:映画型': ['設定状況の魅力', 'ゲーム本来の楽しさ'],
        'C3:現実活動型': ['感覚運動的興奮', 'ゲーム本来の楽しさ'],
        'C4:キャラクター型': ['和みと癒し', 'ゲーム本来の楽しさ'],
        'C5:シミュレーション型': ['難解・頭脳型', '和みと癒し']
    }
    
    common_impression_scores = {}
    
    for game_type, related_emotions in type_emotion_mapping.items():
        # 関連する主観感情の類似度の平均を計算
        related_scores = [emotional_similarities.get(emotion, 0.0) for emotion in related_emotions]
        avg_emotional_score = np.mean(related_scores) if related_scores else 0.0
        
        # 型の適合率
        type_percentage = game_type_percentages.get(game_type, 0.0) / 100.0  # 0-1に正規化
        
        # 共通印象スコア = 主観感情スコア × 型適合率
        common_impression_scores[game_type] = avg_emotional_score * type_percentage
    
    return common_impression_scores

