"""
HMMの出力結果から感情を判定し、対応するゲームタイトルをレコメンドするシステム
"""
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
import os

# HMMで使用される4つの感情状態
HMM_EMOTIONAL_STATES = [
    '感覚運動的興奮',
    '難解・頭脳型',
    '和みと癒し',
    '設定状況の魅力'
]

# DBファイルのパス
DB_DIR = os.path.join(os.path.dirname(__file__))
EMOTION_GAME_MAPPING_CSV = os.path.join(DB_DIR, 'emotion_game_mapping.csv')


def analyze_emotion_distribution(emotion_sequence: List[str]) -> Dict[str, int]:
    """
    HMMの出力結果（感情の時系列リスト）から感情の分布を計算
    
    Args:
        emotion_sequence: 感情ラベルのリスト（例: ['設定状況の魅力', '和みと癒し', ...]）
    
    Returns:
        感情ラベルをキー、出現回数を値とする辞書
    """
    if not emotion_sequence:
        return {}
    
    counter = Counter(emotion_sequence)
    return dict(counter)


def get_dominant_emotion(emotion_sequence: List[str]) -> Optional[str]:
    """
    HMMの出力結果から最も多い感情を判定
    
    Args:
        emotion_sequence: 感情ラベルのリスト
    
    Returns:
        最も多い感情ラベル。データが空の場合はNone
    """
    if not emotion_sequence:
        return None
    
    distribution = analyze_emotion_distribution(emotion_sequence)
    if not distribution:
        return None
    
    # 最も多い感情を返す（同数の場合は最初に出現したものを返す）
    dominant_emotion = max(distribution, key=distribution.get)
    return dominant_emotion


def load_emotion_game_mapping(csv_path: str = EMOTION_GAME_MAPPING_CSV) -> pd.DataFrame:
    """
    感情とゲームタイトルの対応関係を読み込む
    
    Args:
        csv_path: マッピングCSVファイルのパス
    
    Returns:
        感情とゲームタイトルの対応関係を含むDataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"感情-ゲームタイトルマッピングファイルが見つかりません: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def get_games_by_emotion(emotion_label: str, csv_path: str = EMOTION_GAME_MAPPING_CSV) -> List[str]:
    """
    指定された感情ラベルに対応するゲームタイトルのリストを取得
    
    Args:
        emotion_label: 感情ラベル
        csv_path: マッピングCSVファイルのパス
    
    Returns:
        ゲームタイトルのリスト
    """
    df = load_emotion_game_mapping(csv_path)
    
    # 指定された感情ラベルに該当するゲームをフィルタ
    filtered_df = df[df['emotion_label'] == emotion_label]
    
    if filtered_df.empty:
        return []
    
    # 類似度スコアでソート（降順）
    filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
    
    return filtered_df['game_title'].tolist()


def recommend_games(emotion_sequence: List[str], 
                   csv_path: str = EMOTION_GAME_MAPPING_CSV,
                   top_n: int = 5) -> Tuple[Optional[str], List[str], Dict[str, int]]:
    """
    HMMの出力結果から感情を判定し、対応するゲームタイトルをレコメンド
    
    Args:
        emotion_sequence: HMMの出力結果（感情ラベルのリスト）
        csv_path: マッピングCSVファイルのパス
        top_n: レコメンドするゲームの最大数
    
    Returns:
        (dominant_emotion, recommended_games, emotion_distribution)のタプル
        - dominant_emotion: 最も多い感情ラベル
        - recommended_games: レコメンドされるゲームタイトルのリスト
        - emotion_distribution: 感情の分布（キー: 感情ラベル, 値: 出現回数）
    """
    # 感情の分布を計算
    emotion_distribution = analyze_emotion_distribution(emotion_sequence)
    
    # 最も多い感情を判定
    dominant_emotion = get_dominant_emotion(emotion_sequence)
    
    if dominant_emotion is None:
        return None, [], emotion_distribution
    
    # 該当する感情に対応するゲームタイトルを取得
    recommended_games = get_games_by_emotion(dominant_emotion, csv_path)
    
    # top_nで制限
    if top_n > 0:
        recommended_games = recommended_games[:top_n]
    
    return dominant_emotion, recommended_games, emotion_distribution


def update_emotion_game_mapping_from_maegumi(maegumi_csv_path: str,
                                            output_csv_path: str = EMOTION_GAME_MAPPING_CSV):
    """
    MAEgumi.csvから感情とゲームタイトルの対応関係を更新
    「ゲーム本来の楽しさ」に分類されたゲームは、HMMの4つの感情のうち最も類似度が高いものにマッピング
    
    Args:
        maegumi_csv_path: MAEgumi.csvのパス
        output_csv_path: 出力CSVファイルのパス
    """
    df = pd.read_csv(maegumi_csv_path)
    
    results = []
    
    for _, row in df.iterrows():
        game_title = row['game_title']
        estimated_label = row['estimated_emotional_label']
        
        # HMMの4つの感情の類似度を取得
        hmm_emotion_scores = {}
        for emotion in HMM_EMOTIONAL_STATES:
            similarity_col = f'similarity_{emotion}'
            if similarity_col in row:
                hmm_emotion_scores[emotion] = row[similarity_col]
        
        # 「ゲーム本来の楽しさ」に分類された場合は、HMMの4つの感情のうち最も類似度が高いものを使用
        if estimated_label == 'ゲーム本来の楽しさ':
            if hmm_emotion_scores:
                # 最も類似度が高い感情を選択
                mapped_emotion = max(hmm_emotion_scores, key=hmm_emotion_scores.get)
                similarity_score = hmm_emotion_scores[mapped_emotion]
            else:
                # 類似度データがない場合はスキップ
                continue
        elif estimated_label in HMM_EMOTIONAL_STATES:
            # HMMの4つの感情のいずれかに該当する場合
            mapped_emotion = estimated_label
            similarity_col = f'similarity_{estimated_label}'
            if similarity_col in row:
                similarity_score = row[similarity_col]
            else:
                similarity_score = 0.0
        else:
            # その他の感情の場合はスキップ
            continue
        
        results.append({
            'game_title': game_title,
            'emotion_label': mapped_emotion,
            'similarity_score': similarity_score
        })
    
    # DataFrameに変換して保存
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(['emotion_label', 'similarity_score'], ascending=[True, False])
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"感情-ゲームタイトルマッピングを '{output_csv_path}' に保存しました。")
    print(f"合計 {len(result_df)} 件のゲームを登録しました。")
    
    # 感情ごとのゲーム数を表示
    emotion_counts = result_df['emotion_label'].value_counts()
    print("\n感情ごとのゲーム数:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}件")


if __name__ == '__main__':
    # MAEgumi.csvからデータベースを更新
    maegumi_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'gametitleSENTEI',
        '5rabel',
        'MAEgumi.csv'
    )
    
    if os.path.exists(maegumi_path):
        print(f"MAEgumi.csvからデータベースを更新中: {maegumi_path}")
        update_emotion_game_mapping_from_maegumi(maegumi_path)
    else:
        print(f"警告: MAEgumi.csvが見つかりません: {maegumi_path}")
        print("既存のデータベースを使用します。")
    
    # テスト: サンプルの感情シーケンスでレコメンデーション
    test_emotion_sequence = [
        '設定状況の魅力', '設定状況の魅力', '設定状況の魅力',
        '和みと癒し', '和みと癒し',
        '感覚運動的興奮'
    ]
    
    print("\n--- レコメンデーションテスト ---")
    print(f"入力感情シーケンス: {test_emotion_sequence}")
    
    dominant_emotion, recommended_games, emotion_dist = recommend_games(test_emotion_sequence)
    
    print(f"\n判定された主要感情: {dominant_emotion}")
    print(f"感情の分布: {emotion_dist}")
    print(f"レコメンドされるゲーム: {recommended_games}")

