"""
HMM.ipynbの出力結果を処理し、ゲームタイトルをレコメンドするスクリプト
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DB.hmm_recommendation import (
    process_hmm_output_and_recommend,
    print_recommendation_result,
    extract_emotions_from_hmm_output
)

# HMMのモデルとクラスをインポート
HMM_DIR = os.path.join(PROJECT_ROOT, 'HMMleran')
if HMM_DIR not in sys.path:
    sys.path.insert(0, HMM_DIR)

try:
    from HMM.ipynb import HMMGameEmotionClassifier, extract_features_from_keylog
except ImportError:
    # HMM.ipynbから直接インポートできない場合、別の方法で処理
    pass


def process_keylog_and_recommend(keylog_csv_path: str,
                                 model_path: Optional[str] = None,
                                 time_window_sec: float = 30.0,
                                 top_n: int = 5,
                                 output_csv_path: Optional[str] = None) -> Dict:
    """
    キーログCSVから特徴量を抽出し、HMMで感情を推定してゲームをレコメンド
    
    Args:
        keylog_csv_path: キーログCSVファイルのパス
        model_path: HMMモデルファイルのパス（デフォルト: HMMleran/hmm_emotion_model_4states.pkl）
        time_window_sec: 時間窓のサイズ（秒）
        top_n: レコメンドするゲームの最大数
        output_csv_path: 結果を保存するCSVファイルのパス
    
    Returns:
        レコメンデーション結果を含む辞書
    """
    # HMMモデルを読み込み
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'HMMleran', 'hmm_emotion_model_4states.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HMMモデルファイルが見つかりません: {model_path}")
    
    # HMMモデルと特徴量抽出関数をインポート
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'HMMleran'))
    
    # HMMクラスと特徴量抽出関数を動的にインポート
    import importlib.util
    hmm_module_path = os.path.join(PROJECT_ROOT, 'HMMleran', 'HMM.ipynb')
    
    # HMM.ipynbからクラスと関数を読み込む（実際には別ファイルに分離する必要がある）
    # ここでは、既存のHMMの出力結果CSVから感情を取得する方法を使用
    
    raise NotImplementedError(
        "キーログから直接処理する機能は、HMMクラスと特徴量抽出関数を別ファイルに分離する必要があります。"
        "既存のHMMの出力結果CSVファイルを使用してください。"
    )


def process_hmm_emotion_output(hmm_output_csv_path: str,
                               emotion_column: str = '推定感情',
                               top_n: int = 5,
                               output_csv_path: Optional[str] = None) -> Dict:
    """
    HMMの出力結果CSVファイルから感情を読み取り、ゲームをレコメンド
    
    Args:
        hmm_output_csv_path: HMMの出力結果CSVファイルのパス
        emotion_column: 感情ラベルが格納されている列名
        top_n: レコメンドするゲームの最大数
        output_csv_path: 結果を保存するCSVファイルのパス
    
    Returns:
        レコメンデーション結果を含む辞書
    """
    return process_hmm_output_and_recommend(
        hmm_output_path=hmm_output_csv_path,
        emotion_column=emotion_column,
        top_n=top_n,
        output_csv_path=output_csv_path
    )


def process_emotion_sequence(emotion_sequence: List[str],
                            top_n: int = 5,
                            output_csv_path: Optional[str] = None) -> Dict:
    """
    感情ラベルのリストからゲームをレコメンド
    
    Args:
        emotion_sequence: 感情ラベルのリスト
        top_n: レコメンドするゲームの最大数
        output_csv_path: 結果を保存するCSVファイルのパス
    
    Returns:
        レコメンデーション結果を含む辞書
    """
    return process_hmm_output_and_recommend(
        emotion_sequence=emotion_sequence,
        top_n=top_n,
        output_csv_path=output_csv_path
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HMMの出力結果からゲームタイトルをレコメンド',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # HMMの出力結果CSVからレコメンド
  python DB/process_hmm_and_recommend.py --input mizuki2025/来場様用結果/emotion_analysis_outputaikawa.csv
  
  # 感情ラベルのリストからレコメンド（テスト用）
  python DB/process_hmm_and_recommend.py --emotions "設定状況の魅力,設定状況の魅力,和みと癒し"
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='HMMの出力結果CSVファイルのパス')
    parser.add_argument('--emotion-column', '-c', type=str, default='推定感情',
                       help='感情ラベルが格納されている列名（デフォルト: 推定感情）')
    parser.add_argument('--emotions', '-e', type=str,
                       help='感情ラベルのカンマ区切りリスト（テスト用）')
    parser.add_argument('--top-n', '-n', type=int, default=5,
                       help='レコメンドするゲームの最大数（デフォルト: 5）')
    parser.add_argument('--output', '-o', type=str,
                       help='結果を保存するCSVファイルのパス')
    
    args = parser.parse_args()
    
    if args.input:
        # HMMの出力結果CSVファイルから処理
        print(f"HMMの出力結果を読み込み中: {args.input}")
        result = process_hmm_emotion_output(
            hmm_output_csv_path=args.input,
            emotion_column=args.emotion_column,
            top_n=args.top_n,
            output_csv_path=args.output
        )
    elif args.emotions:
        # 感情ラベルのリストから処理
        emotion_list = [e.strip() for e in args.emotions.split(',')]
        print(f"感情ラベルのリストから処理: {emotion_list}")
        result = process_emotion_sequence(
            emotion_sequence=emotion_list,
            top_n=args.top_n,
            output_csv_path=args.output
        )
    else:
        # デフォルト: テスト用のサンプルデータ
        print("テストモード: サンプルの感情シーケンスを使用")
        test_emotion_sequence = [
            '設定状況の魅力', '設定状況の魅力', '設定状況の魅力',
            '和みと癒し', '和みと癒し',
            '感覚運動的興奮'
        ]
        result = process_emotion_sequence(
            emotion_sequence=test_emotion_sequence,
            top_n=args.top_n,
            output_csv_path=args.output
        )
    
    # 結果を表示
    print_recommendation_result(result)

