import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib 
import os
from typing import Dict, List, Tuple, Any

# HMMの隠れ状態（感情ラベル）：4つ
EMOTIONAL_STATES = [
    '感覚運動的興奮', 
    '難解・頭脳型', 
    '和みと癒し', 
    '設定状況の魅力' 
]
N_STATES = len(EMOTIONAL_STATES)
N_FEATURES = 5 # 特徴量の次元数
MODEL_FILENAME = 'hmm_emotion_model_4states.pkl'

# --- 2.2 HMM学習/分類クラス (修正版) ---
class HMMGameEmotionClassifier:
    """キーログ特徴量から4つのゲーム感情ラベルを推定するHMM分類器"""
    
    def __init__(self, n_components: int = N_STATES, n_features: int = N_FEATURES):
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=100,
            init_params="stmc",
            transmat_prior=1.1, # ゼロ頻度対策: 1.0 -> 1.1 に変更 (加算スムージング)
            startprob_prior=1.1 # ゼロ頻度対策: 1.0 -> 1.1 に変更
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
            
            # --- 追加検証: 遷移確率行列が妥当かチェック ---
            if np.any(self.model.transmat_.sum(axis=1) == 0):
                 print("⚠️ 警告: 遷移確率行列にゼロの行が含まれています。データ不足の可能性があります。")
            
            print("✅ HMMモデルの訓練が完了しました。")
        except Exception as e:
            print(f"❌ HMMの訓練中にエラーが発生しました: {e}")

    def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
        """観測データから、最も適合する感情ラベルの時系列を推定します。"""
        # モデルが訓練されているかチェック (簡易的)
        if getattr(self.model, "transmat_", None) is None:
             print("警告: モデルが訓練されていません (transmat_ がありません)。")
             return []
             
        try:
            logprob, state_sequence = self.model.decode(X, algorithm="viterbi")
            return [self.state_names[state] for state in state_sequence]
        except ValueError as e:
            print(f"❌ 推定エラー: {e}")
            return []

    def save_model(self, filename: str = MODEL_FILENAME):
        """訓練済みモデルを保存します。"""
        joblib.dump(self.model, filename)
        joblib.dump(self.state_names, filename.replace('.pkl', '_states.pkl'))
        print(f"✅ モデルと状態名を '{filename}' に保存しました。")

    @classmethod
    def load_model(cls, filename: str = MODEL_FILENAME):
        """保存されたモデルを読み込みます。"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"モデルファイル '{filename}' が見つかりません。")
        instance = cls()
        instance.model = joblib.load(filename)
        instance.state_names = joblib.load(filename.replace('.pkl', '_states.pkl'))
        print(f"✅ モデルを '{filename}' から正常に読み込みました。")
        return instance
