"""
ゲームレコメンデーションシステム - GUIアプリケーション

HMMの出力結果から感情を判定し、ゲームタイトルをレコメンドする統合GUIアプリケーション
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DB.recommendation_system import (
    recommend_games,
    get_dominant_emotion,
    analyze_emotion_distribution,
    update_emotion_game_mapping_from_maegumi,
    EMOTION_GAME_MAPPING_CSV,
    HMM_EMOTIONAL_STATES
)
from DB.hmm_recommendation import (
    process_hmm_output_and_recommend,
    extract_emotions_from_hmm_output,
    save_recommendation_result
)

# HMM関連のインポート
try:
    from hmmlearn import hmm
    import joblib
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("警告: hmmlearnがインストールされていません。キーログからの直接処理は使用できません。")


class GameRecommendationGUI:
    """ゲームレコメンデーションGUIアプリケーション"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ゲームレコメンデーションシステム")
        self.root.geometry("900x700")
        
        # 変数の初期化
        self.hmm_output_path = tk.StringVar()
        self.keylog_path = tk.StringVar()
        self.hmm_model_path = tk.StringVar()
        self.emotion_column = tk.StringVar(value="推定感情")
        self.time_window = tk.DoubleVar(value=30.0)
        self.top_n = tk.IntVar(value=5)
        self.output_path = tk.StringVar()
        
        # デフォルトパスの設定
        self.default_model_path = os.path.join(PROJECT_ROOT, 'HMMleran', 'hmm_emotion_model_4states.pkl')
        if os.path.exists(self.default_model_path):
            self.hmm_model_path.set(self.default_model_path)
        
        # UIの構築
        self.create_widgets()
    
    def create_widgets(self):
        """ウィジェットの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # タイトル
        title_label = ttk.Label(main_frame, text="ゲームレコメンデーションシステム", 
                               font=("Helvetica", 16, "bold"))
        title_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # セパレータ
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, 
                                                             sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 入力ファイル選択セクション
        input_frame = ttk.LabelFrame(main_frame, text="入力ファイル", padding="10")
        input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(1, weight=1)
        row += 1
        
        # HMM出力結果CSV
        ttk.Label(input_frame, text="HMM出力結果CSV:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.hmm_output_path, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.hmm_output_path, "CSVファイルを選択", 
                                                  [("CSV files", "*.csv")])).grid(
            row=0, column=2, pady=5)
        
        # キーログCSV（オプション）
        ttk.Label(input_frame, text="キーログCSV (オプション):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.keylog_path, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.keylog_path, "キーログCSVファイルを選択", 
                                                  [("CSV files", "*.csv")])).grid(
            row=1, column=2, pady=5)
        
        # HMMモデルファイル
        ttk.Label(input_frame, text="HMMモデルファイル:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.hmm_model_path, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.hmm_model_path, "HMMモデルファイルを選択", 
                                                  [("PKL files", "*.pkl")])).grid(
            row=2, column=2, pady=5)
        
        # パラメータ設定セクション
        param_frame = ttk.LabelFrame(main_frame, text="パラメータ設定", padding="10")
        param_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        param_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 感情列名
        ttk.Label(param_frame, text="感情列名:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(param_frame, textvariable=self.emotion_column, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 時間窓サイズ
        ttk.Label(param_frame, text="時間窓サイズ (秒):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(param_frame, from_=10.0, to=300.0, increment=10.0, 
                   textvariable=self.time_window, width=20).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # トップN
        ttk.Label(param_frame, text="レコメンド数 (Top N):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(param_frame, from_=1, to=20, textvariable=self.top_n, width=20).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 出力ファイル
        ttk.Label(param_frame, text="出力CSV (オプション):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(param_frame, textvariable=self.output_path, width=50).grid(
            row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(param_frame, text="参照", 
                  command=lambda: self.select_file(self.output_path, "出力CSVファイルを選択", 
                                                  [("CSV files", "*.csv")], save=True)).grid(
            row=3, column=2, pady=5)
        
        # ボタンセクション
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        ttk.Button(button_frame, text="レコメンデーション実行", 
                  command=self.execute_recommendation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="データベース更新", 
                  command=self.update_database).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="クリア", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # 結果表示セクション
        result_frame = ttk.LabelFrame(main_frame, text="結果", padding="10")
        result_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        # 結果テキストエリア
        self.result_text = scrolledtext.ScrolledText(result_frame, width=80, height=20, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def select_file(self, var: tk.StringVar, title: str, filetypes: List, save: bool = False):
        """ファイル選択ダイアログ"""
        if save:
            file_path = filedialog.asksaveasfilename(title=title, filetypes=filetypes)
        else:
            file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        
        if file_path:
            var.set(file_path)
    
    def clear_all(self):
        """すべての入力と結果をクリア"""
        self.hmm_output_path.set("")
        self.keylog_path.set("")
        self.output_path.set("")
        self.result_text.delete(1.0, tk.END)
        messagebox.showinfo("クリア", "すべての入力と結果をクリアしました。")
    
    def update_database(self):
        """データベースを更新"""
        try:
            maegumi_path = os.path.join(
                PROJECT_ROOT,
                'gametitleSENTEI',
                '5rabel',
                'MAEgumi.csv'
            )
            
            if not os.path.exists(maegumi_path):
                messagebox.showerror("エラー", 
                    f"MAEgumi.csvが見つかりません: {maegumi_path}\n"
                    "ファイルパスを確認してください。")
                return
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "データベースを更新中...\n\n")
            self.root.update()
            
            update_emotion_game_mapping_from_maegumi(maegumi_path)
            
            self.result_text.insert(tk.END, f"✅ データベースの更新が完了しました。\n")
            self.result_text.insert(tk.END, f"ファイル: {EMOTION_GAME_MAPPING_CSV}\n\n")
            
            # データベースの内容を表示
            df = pd.read_csv(EMOTION_GAME_MAPPING_CSV)
            self.result_text.insert(tk.END, "データベースの内容:\n")
            self.result_text.insert(tk.END, df.to_string(index=False))
            self.result_text.insert(tk.END, "\n\n")
            
            messagebox.showinfo("成功", "データベースの更新が完了しました。")
        
        except Exception as e:
            error_msg = f"データベースの更新中にエラーが発生しました: {str(e)}"
            self.result_text.insert(tk.END, f"❌ {error_msg}\n")
            messagebox.showerror("エラー", error_msg)
    
    def execute_recommendation(self):
        """レコメンデーションを実行"""
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "レコメンデーション処理を開始します...\n\n")
            self.root.update()
            
            # 入力の検証
            hmm_output = self.hmm_output_path.get().strip()
            keylog = self.keylog_path.get().strip()
            
            if not hmm_output and not keylog:
                messagebox.showerror("エラー", 
                    "HMM出力結果CSVまたはキーログCSVのいずれかを指定してください。")
                return
            
            result = None
            
            # HMM出力結果から処理
            if hmm_output:
                if not os.path.exists(hmm_output):
                    messagebox.showerror("エラー", f"ファイルが見つかりません: {hmm_output}")
                    return
                
                self.result_text.insert(tk.END, f"HMM出力結果を読み込み中: {hmm_output}\n")
                self.root.update()
                
                result = process_hmm_output_and_recommend(
                    hmm_output_path=hmm_output,
                    emotion_column=self.emotion_column.get(),
                    top_n=self.top_n.get(),
                    output_csv_path=self.output_path.get() if self.output_path.get() else None
                )
            
            # キーログから処理（HMM出力結果が指定されていない場合、または両方指定されている場合はHMM出力結果を優先）
            elif keylog and not hmm_output and HMM_AVAILABLE:
                if not os.path.exists(keylog):
                    messagebox.showerror("エラー", f"ファイルが見つかりません: {keylog}")
                    return
                
                model_path = self.hmm_model_path.get().strip()
                if not model_path or not os.path.exists(model_path):
                    messagebox.showerror("エラー", 
                        f"HMMモデルファイルが見つかりません: {model_path}\n"
                        "HMMモデルファイルのパスを指定してください。")
                    return
                
                self.result_text.insert(tk.END, f"キーログを処理中: {keylog}\n")
                self.result_text.insert(tk.END, f"HMMモデル: {model_path}\n")
                self.result_text.insert(tk.END, f"時間窓サイズ: {self.time_window.get()}秒\n\n")
                self.root.update()
                
                # キーログから感情を推定
                result = self.process_keylog_with_hmm(keylog, model_path)
            
            elif keylog and not hmm_output and not HMM_AVAILABLE:
                messagebox.showerror("エラー", 
                    "キーログからの処理にはhmmlearnライブラリが必要です。\n"
                    "pip install hmmlearn でインストールしてください。")
                return
            
            elif not hmm_output and not keylog:
                messagebox.showerror("エラー", 
                    "HMM出力結果CSVまたはキーログCSVのいずれかを指定してください。")
                return
            
            # 結果を表示
            if result:
                self.display_result(result)
            else:
                messagebox.showerror("エラー", "レコメンデーション処理に失敗しました。")
        
        except Exception as e:
            error_msg = f"レコメンデーション処理中にエラーが発生しました: {str(e)}"
            self.result_text.insert(tk.END, f"❌ {error_msg}\n")
            messagebox.showerror("エラー", error_msg)
            import traceback
            self.result_text.insert(tk.END, traceback.format_exc())
    
    def process_keylog_with_hmm(self, keylog_path: str, model_path: str) -> Dict:
        """キーログからHMMで感情を推定してレコメンド"""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearnライブラリがインストールされていません。")
        
        # キーログデータを読み込み
        self.result_text.insert(tk.END, "キーログデータを読み込み中...\n")
        self.root.update()
        
        keylog_data = pd.read_csv(keylog_path)
        
        # 特徴量を抽出
        self.result_text.insert(tk.END, "特徴量を抽出中...\n")
        self.root.update()
        
        X_test, _ = self.extract_features_from_keylog(keylog_data, self.time_window.get())
        
        if X_test.size == 0:
            raise ValueError("特徴量の抽出に失敗しました。キーログデータの形式を確認してください。")
        
        # HMMモデルを読み込み
        self.result_text.insert(tk.END, "HMMモデルを読み込み中...\n")
        self.root.update()
        
        classifier = self.load_hmm_model(model_path)
        
        # 感情を推定
        self.result_text.insert(tk.END, "感情を推定中...\n")
        self.root.update()
        
        estimated_emotions = classifier.predict_emotion_sequence(X_test)
        
        self.result_text.insert(tk.END, f"感情推定が完了しました。総時間窓数: {len(estimated_emotions)}\n\n")
        self.root.update()
        
        # レコメンデーションを実行
        result = process_hmm_output_and_recommend(
            emotion_sequence=estimated_emotions,
            top_n=self.top_n.get(),
            output_csv_path=self.output_path.get() if self.output_path.get() else None
        )
        
        return result
    
    def extract_features_from_keylog(self, log_data: pd.DataFrame, 
                                    time_window_sec: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def load_hmm_model(self, model_path: str):
        """HMMモデルを読み込み"""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearnライブラリがインストールされていません。")
        
        class HMMGameEmotionClassifier:
            def __init__(self):
                self.model = None
                self.state_names = HMM_EMOTIONAL_STATES
            
            def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
                if self.model is None:
                    raise ValueError("モデルが読み込まれていません。")
                if X.size == 0:
                    return []
                logprob, state_sequence = self.model.decode(X, algorithm="viterbi")
                return [self.state_names[state] for state in state_sequence]
            
            @classmethod
            def load_model(cls, filename: str):
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"モデルファイルが見つかりません: {filename}")
                instance = cls()
                instance.model = joblib.load(filename)
                state_file = filename.replace('.pkl', '_states.pkl')
                if os.path.exists(state_file):
                    instance.state_names = joblib.load(state_file)
                else:
                    # 状態ファイルがない場合はデフォルトの状態名を使用
                    instance.state_names = HMM_EMOTIONAL_STATES
                return instance
        
        return HMMGameEmotionClassifier.load_model(model_path)
    
    def display_result(self, result: Dict):
        """結果を表示"""
        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "レコメンデーション結果\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")
        
        # 判定された主要感情
        self.result_text.insert(tk.END, "【判定された主要感情】\n")
        self.result_text.insert(tk.END, f"  {result['dominant_emotion']}\n\n")
        
        # 感情の分布
        self.result_text.insert(tk.END, "【感情の分布】\n")
        self.result_text.insert(tk.END, f"  総感情数: {result['total_emotions']}\n")
        for emotion, count in result['emotion_distribution'].items():
            percentage = result['emotion_percentages'].get(emotion, 0.0)
            self.result_text.insert(tk.END, f"  {emotion}: {count}回 ({percentage}%)\n")
        self.result_text.insert(tk.END, "\n")
        
        # レコメンドされるゲーム
        self.result_text.insert(tk.END, "【レコメンドされるゲーム】\n")
        if result['recommended_games']:
            for i, game in enumerate(result['recommended_games'], 1):
                self.result_text.insert(tk.END, f"  {i}. {game}\n")
        else:
            self.result_text.insert(tk.END, "  レコメンドされるゲームが見つかりませんでした。\n")
        
        self.result_text.insert(tk.END, "="*60 + "\n\n")
        
        # 出力ファイルが指定されている場合
        if self.output_path.get():
            self.result_text.insert(tk.END, f"結果を '{self.output_path.get()}' に保存しました。\n")


def main():
    """メイン関数"""
    root = tk.Tk()
    app = GameRecommendationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

