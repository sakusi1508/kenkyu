"""
HMM.ipynbから感情推定結果を取得し、ゲームをレコメンドする統合スクリプト

このスクリプトは、HMM.ipynbのコードを利用してキーログから感情を推定し、
レコメンデーションシステムに統合します。
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DB.hmm_recommendation import (
    process_hmm_output_and_recommend,
    print_recommendation_result
)

# HMMのクラスと関数をインポート
HMM_DIR = os.path.join(PROJECT_ROOT, 'HMMleran')
if HMM_DIR not in sys.path:
    sys.path.insert(0, HMM_DIR)

# HMMのクラス定義をインポート（HMM.ipynbから）
try:
    # HMM.ipynbから直接インポートできないため、必要な関数とクラスをここに定義
    from hmmlearn import hmm
    import joblib
    
    # HMMの感情状態
    EMOTIONAL_STATES = [
        '感覚運動的興奮',
        '難解・頭脳型',
        '和みと癒し',
        '設定状況の魅力'
    ]
    
    # 特徴量抽出関数（HMM.ipynbから）
    def extract_features_from_keylog(log_data: pd.DataFrame, time_window_sec: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """キーログデータから特徴量を抽出"""
        log_data.rename(columns={
            '経過時間(s) (セッション開始からの時間)': 'elapsed_time',
            '持続時間(s)': 'duration',
            '遅延時間(s)': 'delay'
        }, inplace=True)
        
        try:
            log_data['elapsed_time'] = log_data['elapsed_time'].astype(float)
            log_data['duration'] = log_data['duration'].astype(float)
            log_data['delay'] = log_data['delay'].astype(float)
        except Exception as e:
            print(f"データの型変換エラー: {e}")
            return np.array([]), np.array([])
        
        max_time = log_data['elapsed_time'].max()
        segments = []
        
        start_time = 0.0
        while start_time < max_time + time_window_sec:
            end_time = start_time + time_window_sec
            window_data = log_data[(log_data['elapsed_time'] >= start_time) & (log_data['elapsed_time'] < end_time)]
            
            if len(window_data) == 0:
                feature_vector = [0.0, time_window_sec, 0.0, 0.0, 1.0]
            else:
                duration_mean = window_data['duration'].mean()
                delay_mean = window_data['delay'].mean()
                apm = len(window_data) / time_window_sec * 60
                duration_var = window_data['duration'].var()
                if np.isnan(duration_var):
                    duration_var = 0.0
                total_active_duration = window_data['duration'].sum()
                stop_time_ratio = (time_window_sec - total_active_duration) / time_window_sec
                stop_time_ratio = max(0.0, min(1.0, stop_time_ratio))
                
                feature_vector = [duration_mean, delay_mean, apm, duration_var, stop_time_ratio]
            
            segments.append(feature_vector)
            start_time = end_time
        
        X = np.array(segments)
        lengths = np.array([len(X)])
        
        return X, lengths
    
    # HMM分類器クラス（HMM.ipynbから）
    class HMMGameEmotionClassifier:
        """キーログ特徴量から4つのゲーム感情ラベルを推定するHMM分類器"""
        
        def __init__(self, n_components: int = 4, n_features: int = 5):
            self.model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=100,
                init_params="stmc"
            )
            self.state_names = EMOTIONAL_STATES
        
        def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
            """観測データから、最も適合する感情ラベルの時系列を推定"""
            if self.model.n_iter == 0:
                print("警告: モデルが訓練されていません。")
                return []
            logprob, state_sequence = self.model.decode(X, algorithm="viterbi")
            return [self.state_names[state] for state in state_sequence]
        
        @classmethod
        def load_model(cls, filename: str):
            """保存されたモデルを読み込み"""
            if not os.path.exists(filename):
                raise FileNotFoundError(f"モデルファイル '{filename}' が見つかりません。")
            instance = cls()
            instance.model = joblib.load(filename)
            state_file = filename.replace('.pkl', '_states.pkl')
            if os.path.exists(state_file):
                instance.state_names = joblib.load(state_file)
            else:
                instance.state_names = EMOTIONAL_STATES
            print(f"✅ モデルを '{filename}' から正常に読み込みました。")
            return instance

except ImportError as e:
    print(f"警告: HMM関連のライブラリをインポートできませんでした: {e}")
    print("HMMの出力結果CSVファイルから直接処理してください。")


def process_keylog_with_hmm(keylog_csv_path: str,
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
        raise FileNotFoundError(
            f"HMMモデルファイルが見つかりません: {model_path}\n"
            "HMM.ipynbを実行してモデルを訓練・保存してください。"
        )
    
    # キーログデータを読み込み
    print(f"キーログデータを読み込み中: {keylog_csv_path}")
    try:
        keylog_data = pd.read_csv(keylog_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"キーログファイルが見つかりません: {keylog_csv_path}")
    
    # 特徴量を抽出
    print(f"特徴量を抽出中（時間窓: {time_window_sec}秒）...")
    X_test, _ = extract_features_from_keylog(keylog_data, time_window_sec=time_window_sec)
    
    if X_test.size == 0:
        raise ValueError("特徴量の抽出に失敗しました。キーログデータの形式を確認してください。")
    
    # HMMモデルを読み込み
    print(f"HMMモデルを読み込み中: {model_path}")
    classifier = HMMGameEmotionClassifier.load_model(model_path)
    
    # 感情を推定
    print("感情を推定中...")
    estimated_emotions = classifier.predict_emotion_sequence(X_test)
    
    print(f"感情推定が完了しました。総時間窓数: {len(estimated_emotions)}")
    
    # レコメンデーションを実行
    result = process_hmm_output_and_recommend(
        emotion_sequence=estimated_emotions,
        top_n=top_n,
        output_csv_path=output_csv_path
    )
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='キーログからHMMで感情を推定し、ゲームをレコメンド',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # キーログから直接処理
  python DB/integrate_hmm_recommendation.py --keylog keydata/keylog_output.csv
  
  # HMMモデルのパスを指定
  python DB/integrate_hmm_recommendation.py --keylog keydata/keylog_output.csv --model HMMleran/hmm_emotion_model_4states.pkl
        """
    )
    
    parser.add_argument('--keylog', '-k', type=str,
                       help='キーログCSVファイルのパス')
    parser.add_argument('--model', '-m', type=str,
                       help='HMMモデルファイルのパス（デフォルト: HMMleran/hmm_emotion_model_4states.pkl）')
    parser.add_argument('--time-window', '-t', type=float, default=30.0,
                       help='時間窓のサイズ（秒、デフォルト: 30.0）')
    parser.add_argument('--top-n', '-n', type=int, default=5,
                       help='レコメンドするゲームの最大数（デフォルト: 5）')
    parser.add_argument('--output', '-o', type=str,
                       help='結果を保存するCSVファイルのパス')
    
    args = parser.parse_args()
    
    if args.keylog:
        # キーログから直接処理
        result = process_keylog_with_hmm(
            keylog_csv_path=args.keylog,
            model_path=args.model,
            time_window_sec=args.time_window,
            top_n=args.top_n,
            output_csv_path=args.output
        )
        
        # 結果を表示
        print_recommendation_result(result)
    else:
        parser.print_help()
        print("\nエラー: --keylogオプションでキーログファイルのパスを指定してください。")

