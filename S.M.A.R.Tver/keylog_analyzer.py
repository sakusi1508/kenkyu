"""
キーログから14因子を推定し、4個の主観感情を推定するモジュール
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from hmmlearn import hmm
    import joblib
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# ローカルモジュールをインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_converter import (
    FACTOR_LIST, convert_to_emotional_labels, convert_to_game_types,
    compare_emotional_and_type_similarity
)


def extract_features_from_keylog(log_data: pd.DataFrame, time_window_sec: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    キーログデータから特徴量を抽出
    
    Args:
        log_data: キーログデータのDataFrame
        time_window_sec: 時間窓のサイズ（秒）
    
    Returns:
        (特徴量配列, 長さ配列)のタプル
    """
    # 列名の統一
    log_data = log_data.copy()
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
        raise ValueError(f"データの型変換エラー: {e}")
    
    max_time = log_data['elapsed_time'].max()
    segments = []
    
    start_time = 0.0
    while start_time < max_time + time_window_sec:
        end_time = start_time + time_window_sec
        window_data = log_data[(log_data['elapsed_time'] >= start_time) & 
                              (log_data['elapsed_time'] < end_time)]
        
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


def estimate_factors_from_keylog_features(features: np.ndarray) -> Dict[str, float]:
    """
    キーログの特徴量から14因子を推定
    
    注意: これは簡易的な推定方法です。実際の実装では、機械学習モデルを使用することを推奨します。
    
    Args:
        features: キーログの特徴量配列（各時間窓の特徴量）
    
    Returns:
        14因子の値を持つ辞書
    """
    # 特徴量の平均を計算
    if features.size == 0:
        # デフォルト値
        return {factor: 0.5 for factor in FACTOR_LIST}
    
    avg_features = np.mean(features, axis=0) if len(features.shape) > 1 else features
    
    # 簡易的なマッピング（実際の実装では、より高度なモデルを使用）
    # [duration_mean, delay_mean, apm, duration_var, stop_time_ratio]
    duration_mean = avg_features[0] if len(avg_features) > 0 else 0.0
    delay_mean = avg_features[1] if len(avg_features) > 1 else 0.0
    apm = avg_features[2] if len(avg_features) > 2 else 0.0
    duration_var = avg_features[3] if len(avg_features) > 3 else 0.0
    stop_time_ratio = avg_features[4] if len(avg_features) > 4 else 0.0
    
    # 14因子への簡易マッピング（0-1の範囲に正規化）
    factors = {
        'RPG性': min(1.0, max(0.0, (1.0 - stop_time_ratio) * 0.7)),
        'アクション性': min(1.0, max(0.0, apm / 300.0)),  # APMが高いほどアクション性が高い
        'パズル性': min(1.0, max(0.0, duration_mean * 2.0)),  # 持続時間が長いほどパズル性が高い
        '報酬の程度': min(1.0, max(0.0, (1.0 - stop_time_ratio) * 0.6)),
        'グラフィックのリアル性': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        '音響効果': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        '世界観の仮想・現実性': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        'キャラクターの知名度': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        '文化の異質・同質感': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        '操作のリアクション': min(1.0, max(0.0, (1.0 - delay_mean) * 2.0)),  # 遅延が少ないほど反応が良い
        '遂行時間の長さ': min(1.0, max(0.0, (1.0 - stop_time_ratio) * 0.8)),
        'シナリオの重要度': min(1.0, max(0.0, duration_mean * 1.5)),
        '参加人数の多さ': min(1.0, max(0.0, 0.5)),  # キーログからは推定困難
        'ハードウェア普及度': min(1.0, max(0.0, 0.5))  # キーログからは推定困難
    }
    
    return factors


def estimate_emotions_from_keylog(
    keylog_path: str,
    hmm_model_path: Optional[str] = None,
    time_window_sec: float = 30.0
) -> Dict[str, any]:
    """
    キーログから4個の主観感情を推定
    
    Args:
        keylog_path: キーログCSVファイルのパス
        hmm_model_path: HMMモデルファイルのパス（オプション）
        time_window_sec: 時間窓のサイズ（秒）
    
    Returns:
        推定結果を含む辞書
    """
    # キーログデータを読み込み
    keylog_data = pd.read_csv(keylog_path)
    
    # 方法1: HMMモデルを使用して感情を推定（推奨）
    if hmm_model_path and HMM_AVAILABLE and os.path.exists(hmm_model_path):
        # 特徴量を抽出
        X, lengths = extract_features_from_keylog(keylog_data, time_window_sec)
        
        # HMMモデルを読み込み
        try:
            model = joblib.load(hmm_model_path)
            state_file = hmm_model_path.replace('.pkl', '_states.pkl')
            if os.path.exists(state_file):
                state_names = joblib.load(state_file)
            else:
                # デフォルトの状態名
                from DB.recommendation_system import HMM_EMOTIONAL_STATES
                state_names = HMM_EMOTIONAL_STATES
            
            # 感情を推定
            logprob, state_sequence = model.decode(X, algorithm="viterbi")
            estimated_emotions = [state_names[state] for state in state_sequence]
            
            # 最も多い感情を取得
            from collections import Counter
            emotion_counter = Counter(estimated_emotions)
            dominant_emotion = emotion_counter.most_common(1)[0][0]
            
            # 14因子を推定（簡易的な方法）
            factors = estimate_factors_from_keylog_features(X)
            
        except Exception as e:
            print(f"HMMモデルの読み込みに失敗しました: {e}")
            # フォールバック: 特徴量から直接推定
            X, lengths = extract_features_from_keylog(keylog_data, time_window_sec)
            factors = estimate_factors_from_keylog_features(X)
            dominant_emotion = None
            estimated_emotions = []
    
    else:
        # 方法2: 特徴量から直接14因子を推定
        X, lengths = extract_features_from_keylog(keylog_data, time_window_sec)
        factors = estimate_factors_from_keylog_features(X)
        dominant_emotion = None
        estimated_emotions = []
    
    # 14因子から5個の主観感情への変換
    emotional_similarities = convert_to_emotional_labels(factors)
    
    # 14因子から5個の型への変換
    game_type_percentages = convert_to_game_types(factors)
    
    # 共通印象の比較
    common_impression_scores = compare_emotional_and_type_similarity(
        emotional_similarities, game_type_percentages
    )
    
    return {
        'factors': factors,
        'emotional_similarities': emotional_similarities,
        'game_type_percentages': game_type_percentages,
        'common_impression_scores': common_impression_scores,
        'dominant_emotion': dominant_emotion,
        'estimated_emotions': estimated_emotions
    }

