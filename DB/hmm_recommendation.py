"""
HMMの出力結果から感情を判定し、ゲームタイトルをレコメンドする統合システム
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Optional, Tuple
from collections import Counter

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DB.recommendation_system import (
    recommend_games,
    get_dominant_emotion,
    analyze_emotion_distribution,
    load_emotion_game_mapping,
    EMOTION_GAME_MAPPING_CSV
)

# HMMで使用される4つの感情状態
HMM_EMOTIONAL_STATES = [
    '感覚運動的興奮',
    '難解・頭脳型',
    '和みと癒し',
    '設定状況の魅力'
]


def extract_emotions_from_hmm_output(csv_path: str, emotion_column: str = '推定感情') -> List[str]:
    """
    HMMの出力結果CSVから感情ラベルのリストを抽出
    
    Args:
        csv_path: HMMの出力結果CSVファイルのパス
        emotion_column: 感情ラベルが格納されている列名
    
    Returns:
        感情ラベルのリスト
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"HMMの出力結果ファイルが見つかりません: {csv_path}")
    
    if emotion_column not in df.columns:
        raise ValueError(f"感情ラベルの列 '{emotion_column}' が見つかりません。利用可能な列: {list(df.columns)}")
    
    # 感情ラベルのリストを取得（NaNを除外）
    emotions = df[emotion_column].dropna().tolist()
    
    # HMMの4つの感情状態に含まれない感情を除外
    valid_emotions = [emotion for emotion in emotions if emotion in HMM_EMOTIONAL_STATES]
    
    return valid_emotions


def extract_emotions_from_emotion_sequence(emotion_sequence: List[str]) -> List[str]:
    """
    感情ラベルのリストから有効な感情のみを抽出
    
    Args:
        emotion_sequence: 感情ラベルのリスト
    
    Returns:
        有効な感情ラベルのリスト
    """
    valid_emotions = [emotion for emotion in emotion_sequence if emotion in HMM_EMOTIONAL_STATES]
    return valid_emotions


def process_hmm_output_and_recommend(hmm_output_path: Optional[str] = None,
                                     emotion_sequence: Optional[List[str]] = None,
                                     emotion_column: str = '推定感情',
                                     top_n: int = 5,
                                     output_csv_path: Optional[str] = None) -> Dict:
    """
    HMMの出力結果を処理し、ゲームタイトルをレコメンド
    
    Args:
        hmm_output_path: HMMの出力結果CSVファイルのパス（オプション）
        emotion_sequence: 感情ラベルのリスト（オプション、hmm_output_pathが指定されていない場合に使用）
        emotion_column: 感情ラベルが格納されている列名
        top_n: レコメンドするゲームの最大数
        output_csv_path: 結果を保存するCSVファイルのパス（オプション）
    
    Returns:
        レコメンデーション結果を含む辞書
    """
    # 感情シーケンスを取得
    if hmm_output_path:
        emotion_sequence = extract_emotions_from_hmm_output(hmm_output_path, emotion_column)
    elif emotion_sequence:
        emotion_sequence = extract_emotions_from_emotion_sequence(emotion_sequence)
    else:
        raise ValueError("hmm_output_pathまたはemotion_sequenceのいずれかを指定してください。")
    
    if not emotion_sequence:
        raise ValueError("有効な感情データが見つかりません。")
    
    # レコメンデーションを実行
    dominant_emotion, recommended_games, emotion_distribution = recommend_games(
        emotion_sequence,
        top_n=top_n
    )
    
    # 結果をまとめる
    result = {
        'dominant_emotion': dominant_emotion,
        'recommended_games': recommended_games,
        'emotion_distribution': emotion_distribution,
        'total_emotions': len(emotion_sequence),
        'emotion_percentages': {}
    }
    
    # 感情のパーセンテージを計算
    if emotion_distribution:
        total = sum(emotion_distribution.values())
        for emotion, count in emotion_distribution.items():
            result['emotion_percentages'][emotion] = round((count / total) * 100, 2)
    
    # 結果をCSVに保存（オプション）
    if output_csv_path:
        save_recommendation_result(result, output_csv_path)
    
    return result


def save_recommendation_result(result: Dict, output_csv_path: str):
    """
    レコメンデーション結果をCSVファイルに保存
    
    Args:
        result: レコメンデーション結果の辞書
        output_csv_path: 出力CSVファイルのパス
    """
    # 結果をDataFrameに変換
    data = {
        'dominant_emotion': [result['dominant_emotion']],
        'recommended_games': [', '.join(result['recommended_games'])],
        'total_emotions': [result['total_emotions']]
    }
    
    # 感情の分布を追加
    for emotion in HMM_EMOTIONAL_STATES:
        count = result['emotion_distribution'].get(emotion, 0)
        percentage = result['emotion_percentages'].get(emotion, 0.0)
        data[f'{emotion}_count'] = [count]
        data[f'{emotion}_percentage'] = [percentage]
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"レコメンデーション結果を '{output_csv_path}' に保存しました。")


def print_recommendation_result(result: Dict):
    """
    レコメンデーション結果を読みやすい形式で表示
    
    Args:
        result: レコメンデーション結果の辞書
    """
    print("\n" + "="*60)
    print("レコメンデーション結果")
    print("="*60)
    
    print(f"\n【判定された主要感情】")
    print(f"  {result['dominant_emotion']}")
    
    print(f"\n【感情の分布】")
    print(f"  総感情数: {result['total_emotions']}")
    for emotion, count in result['emotion_distribution'].items():
        percentage = result['emotion_percentages'].get(emotion, 0.0)
        print(f"  {emotion}: {count}回 ({percentage}%)")
    
    print(f"\n【レコメンドされるゲーム】")
    if result['recommended_games']:
        for i, game in enumerate(result['recommended_games'], 1):
            print(f"  {i}. {game}")
    else:
        print("  レコメンドされるゲームが見つかりませんでした。")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HMMの出力結果からゲームタイトルをレコメンド')
    parser.add_argument('--input', '-i', type=str, help='HMMの出力結果CSVファイルのパス')
    parser.add_argument('--emotion-column', '-c', type=str, default='推定感情',
                       help='感情ラベルが格納されている列名（デフォルト: 推定感情）')
    parser.add_argument('--top-n', '-n', type=int, default=5,
                       help='レコメンドするゲームの最大数（デフォルト: 5）')
    parser.add_argument('--output', '-o', type=str, help='結果を保存するCSVファイルのパス')
    
    args = parser.parse_args()
    
    if not args.input:
        # テスト用: サンプルの感情シーケンス
        print("テストモード: サンプルの感情シーケンスを使用")
        test_emotion_sequence = [
            '設定状況の魅力', '設定状況の魅力', '設定状況の魅力',
            '和みと癒し', '和みと癒し',
            '感覚運動的興奮'
        ]
        
        result = process_hmm_output_and_recommend(
            emotion_sequence=test_emotion_sequence,
            top_n=args.top_n,
            output_csv_path=args.output
        )
    else:
        # HMMの出力結果CSVファイルから処理
        result = process_hmm_output_and_recommend(
            hmm_output_path=args.input,
            emotion_column=args.emotion_column,
            top_n=args.top_n,
            output_csv_path=args.output
        )
    
    # 結果を表示
    print_recommendation_result(result)

