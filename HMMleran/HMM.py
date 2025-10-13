import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib 
import os
from typing import Dict, List, Tuple, Any

# --- 1. 定数と感情ラベルの定義 ---

# HMMの隠れ状態（感情ラベル）：4つに定義
EMOTIONAL_STATES = [
    '感覚運動的興奮 (C3:現実活動型)', 
    '難解・頭脳型 (C5:シミュレーション型)', 
    '和みと癒し (C1:ファンタジー型)', 
    '設定状況の魅力 (C2:映画型)' 
]
N_STATES = len(EMOTIONAL_STATES)
N_FEATURES = 5 # 特徴量の次元数

# --- 2. キーログからの特徴量抽出関数 ---

def extract_features_from_keylog(log_data: pd.DataFrame, time_window_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    キーログデータ（CSV読み込み後）を時間窓で区切り、HMMの観測特徴量を抽出します。
    
    5つの特徴量: [持続時間平均, 遅延時間平均, APM, 持続時間分散, 停止時間割合]
    """
    
    # 列名の統一と型変換
    log_data.rename(columns={'経過時間(s) (セッション開始からの時間)': 'elapsed_time',
                             '持続時間(s)': 'duration',
                             '遅延時間(s)': 'delay'}, inplace=True)
    
    # データ型が適切かチェックし、変換
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
            # 完全に操作がないウィンドウ: 停止時間100%、その他0
            feature_vector = [0.0, time_window_sec, 0.0, 0.0, 1.0] 
        else:
            # 5つの特徴量の計算
            duration_mean = window_data['duration'].mean()
            delay_mean = window_data['delay'].mean()
            apm = len(window_data) / time_window_sec * 60
            duration_var = window_data['duration'].var()
            if np.isnan(duration_var): duration_var = 0.0
            
            total_active_duration = window_data['duration'].sum()
            stop_time_ratio = (time_window_sec - total_active_duration) / time_window_sec
            stop_time_ratio = max(0.0, min(1.0, stop_time_ratio)) # 0から1にクリップ
            
            feature_vector = [duration_mean, delay_mean, apm, duration_var, stop_time_ratio]
            
        segments.append(feature_vector)
        start_time = end_time

    X = np.array(segments)
    lengths = np.array([len(X)])
    
    return X, lengths

# --- 3. HMM学習/分類クラス ---

class HMMGameEmotionClassifier:
    """キーログ特徴量から4つのゲーム感情ラベルを推定するHMM分類器"""
    
    def __init__(self, n_components: int = N_STATES, n_features: int = N_FEATURES):
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=100,
            init_params="stmc"
        )
        self.state_names = EMOTIONAL_STATES

    def train(self, X: np.ndarray, lengths: np.ndarray):
        """HMMモデルを訓練します。"""
        print(f"HMMモデルを訓練中... (隠れ状態数: {self.model.n_components})")
        
        if X.shape[1] != N_FEATURES:
            print(f"❌ エラー: 訓練データの特徴量次元が一致しません ({X.shape[1]} != {N_FEATURES})")
            return

        try:
            self.model.fit(X, lengths)
            print("✅ HMMモデルの訓練が完了しました。")
        except Exception as e:
            print(f"❌ HMMの訓練中にエラーが発生しました: {e}")

    def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
        """観測データから、最も適合する感情ラベルの時系列を推定します。"""
        if self.model.n_iter == 0:
             print("警告: モデルが訓練されていません。")
             return []
        
        # ビタビアルゴリズムで最適パス（感情の遷移）を計算
        logprob, state_sequence = self.model.decode(X, algorithm="viterbi")
        
        # 状態インデックスを感情ラベル名に変換
        emotion_sequence = [self.state_names[state] for state in state_sequence]
        
        return emotion_sequence

    def save_model(self, filename: str = 'hmm_emotion_model_4states.pkl'):
        """訓練済みモデルを保存します。"""
        joblib.dump(self.model, filename)
        joblib.dump(self.state_names, filename.replace('.pkl', '_states.pkl'))
        print(f"✅ モデルと状態名を '{filename}' に保存しました。")

    @classmethod
    def load_model(cls, filename: str = 'hmm_emotion_model_4states.pkl'):
        """保存されたモデルを読み込みます。"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"モデルファイル '{filename}' が見つかりません。")
        
        instance = cls()
        instance.model = joblib.load(filename)
        instance.state_names = joblib.load(filename.replace('.pkl', '_states.pkl'))
        print(f"✅ モデルを '{filename}' から正常に読み込みました。")
        return instance

# --- 4. 実行ブロック (メイン処理) ---

if __name__ == "__main__":
    
    # ----------------------------------------------------------------------
    # 【 HMM訓練データの準備（デモ用）】
    # 実際には、複数のアノテーション済みキーログCSVを読み込み、
    # それらを結合してX_train (特徴量) と lengths_train (セッション長) を作成する必要があります。
    # ----------------------------------------------------------------------
    
    print("--- HMMゲーム感情推定モデルの実行 ---")
    
    np.random.seed(42)
    
    # 訓練用ダミーデータ生成 (4つの感情状態をシミュレート)
    # X1: 感覚運動的興奮, X2: 難解・頭脳型, X3: 和みと癒し, X4: 設定状況の魅力
    X_excitement = np.abs(np.random.normal(loc=[0.1, 0.05, 100, 0.01, 0.2], scale=0.05, size=(30, 5)))
    X_intellectual = np.abs(np.random.normal(loc=[0.5, 0.8, 10, 0.15, 0.7], scale=0.1, size=(40, 5)))
    X_calmness = np.abs(np.random.normal(loc=[0.3, 0.1, 40, 0.05, 0.4], scale=0.03, size=(30, 5)))
    X_attractive = np.abs(np.random.normal(loc=[0.2, 0.3, 20, 0.08, 0.6], scale=0.08, size=(20, 5)))

    X_train = np.vstack([X_excitement, X_intellectual, X_calmness, X_attractive])
    lengths_train = np.array([len(X_excitement), len(X_intellectual), len(X_calmness), len(X_attractive)])

    # 1. モデルの訓練と保存
    classifier = HMMGameEmotionClassifier()
    classifier.train(X_train, lengths_train)
    model_filename = 'hmm_game_emotion_model_4states.pkl'
    classifier.save_model(model_filename)
    
    # 2. 新しいセッションの推定（テストデータ）
    
    # 例: 興奮型から難解・頭脳型へ遷移する新しいキーログセッション
    X_test_part1 = np.abs(np.random.normal(loc=[0.1, 0.05, 100, 0.01, 0.2], scale=0.05, size=(10, 5)))
    X_test_part2 = np.abs(np.random.normal(loc=[0.5, 0.8, 10, 0.15, 0.7], scale=0.1, size=(10, 5)))
    X_test_new = np.vstack([X_test_part1, X_test_part2])

    # 3. モデルの読み込みと感情推定
    loaded_classifier = HMMGameEmotionClassifier.load_model(model_filename)
    estimated_emotions = loaded_classifier.predict_emotion_sequence(X_test_new)
    
    print("\n--- 最終的な感情推定結果（2秒間隔） ---")
    for i, emotion in enumerate(estimated_emotions):
        time_start = i * 2
        time_end = (i + 1) * 2
        print(f"時間窓 {time_start}-{time_end}秒: {emotion}")