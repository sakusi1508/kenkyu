#!/usr/bin/env python3
"""
感情分析アプリケーション（キーロガー機能なし版）
権限問題を回避し、CSVデータの分析に特化
"""

import time
import csv
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
import joblib

# --- 定数定義 ---
CSV_FILENAME = "emotion_analysis_output.csv"
EMO_CSV_FILENAME = "EMOoutput.csv"

# モデルファイルのパスを設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODEL_FILENAME = os.path.join(project_root, 'HMMleran', 'hmm_emotion_model_4states.pkl')
STATES_FILENAME = os.path.join(project_root, 'HMMleran', 'hmm_emotion_model_4states_states.pkl')

# HMMの隠れ状態（感情ラベル）：4つ
EMOTIONAL_STATES = [
    '感覚運動的興奮', 
    '難解・頭脳型', 
    '和みと癒し', 
    '設定状況の魅力' 
]
N_STATES = len(EMOTIONAL_STATES)
N_FEATURES = 5  # 特徴量の次元数

class EmotionAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("感情分析アプリケーション")
        master.geometry("1000x700")
        
        # データ関連
        self.analysis_data = []
        self.current_emotion = "未分析"
        
        # HMMモデル関連
        self.hmm_model = None
        self.state_names = None
        self.emotion_history = []
        
        # GUIの構築
        self.create_widgets()
        
        # HMMモデルの読み込み
        self.load_hmm_model()
        
        # 終了時のイベントハンドラを設定
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """GUIウィジェットを作成"""
        # メインフレーム
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部フレーム（コントロールパネル）
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # タイトル
        title_label = tk.Label(control_frame, text="感情分析アプリケーション", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 説明文
        info_label = tk.Label(control_frame, 
                             text="CSVファイルからキーログデータを読み込んで感情分析を実行します", 
                             font=("Arial", 10), fg="blue")
        info_label.pack(pady=(0, 10))
        
        # ボタンフレーム
        button_frame = tk.Frame(control_frame)
        button_frame.pack()
        
        # CSV読み込みボタン
        self.import_button = tk.Button(button_frame, text="CSV読み込み", 
                                     command=self.import_csv, 
                                     width=15, bg='lightblue')
        self.import_button.pack(side=tk.LEFT, padx=5)
        
        # 感情分析ボタン
        self.analyze_button = tk.Button(button_frame, text="感情分析実行", 
                                       command=self.analyze_emotions, 
                                       width=15, bg='lightgreen')
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        # CSV保存ボタン
        self.export_button = tk.Button(button_frame, text="分析結果保存", 
                                     command=self.export_results, 
                                     width=15, bg='lightyellow')
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # クリアボタン
        self.clear_button = tk.Button(button_frame, text="クリア", 
                                     command=self.clear_display, 
                                     width=15, bg='lightcoral')
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 現在の感情表示
        emotion_frame = tk.Frame(control_frame)
        emotion_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(emotion_frame, text="現在の感情:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.current_emotion_label = tk.Label(emotion_frame, text=self.current_emotion, 
                                            font=("Arial", 12), fg="red")
        self.current_emotion_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # メインコンテンツエリア（タブ）
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # データ表示タブ
        self.data_frame = tk.Frame(notebook)
        notebook.add(self.data_frame, text="読み込みデータ")
        
        self.data_display = scrolledtext.ScrolledText(self.data_frame, state='disabled', 
                                                   wrap='word', width=100, height=20, 
                                                   font=("Courier", 10))
        self.data_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 感情分析結果タブ
        self.emotion_frame = tk.Frame(notebook)
        notebook.add(self.emotion_frame, text="感情分析結果")
        
        self.emotion_display = scrolledtext.ScrolledText(self.emotion_frame, state='disabled', 
                                                       wrap='word', width=100, height=20, 
                                                       font=("Courier", 10))
        self.emotion_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 統計情報タブ
        self.stats_frame = tk.Frame(notebook)
        notebook.add(self.stats_frame, text="統計情報")
        
        self.stats_display = scrolledtext.ScrolledText(self.stats_frame, state='disabled', 
                                                    wrap='word', width=100, height=20, 
                                                    font=("Courier", 10))
        self.stats_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_hmm_model(self):
        """HMMモデルを読み込む"""
        try:
            if os.path.exists(MODEL_FILENAME) and os.path.exists(STATES_FILENAME):
                self.hmm_model = joblib.load(MODEL_FILENAME)
                self.state_names = joblib.load(STATES_FILENAME)
                self.update_emotion_display("✅ HMMモデルが正常に読み込まれました。")
                print(f"モデルファイル: {MODEL_FILENAME}")
                print(f"状態ファイル: {STATES_FILENAME}")
            else:
                self.update_emotion_display("⚠️ HMMモデルが見つかりません。先にモデルを訓練してください。")
                print(f"モデルファイル存在確認: {os.path.exists(MODEL_FILENAME)}")
                print(f"状態ファイル存在確認: {os.path.exists(STATES_FILENAME)}")
        except Exception as e:
            self.update_emotion_display(f"❌ HMMモデルの読み込みエラー: {e}")
            print(f"HMMモデル読み込みエラー: {e}")

    def import_csv(self):
        """CSVファイルを読み込み"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="CSVファイルを選択"
        )
        
        if filename:
            try:
                df = pd.read_csv(filename)
                self.analysis_data = []
                
                # データの前処理
                for _, row in df.iterrows():
                    data_entry = {
                        "key_info": str(row.get("キー情報", "")),
                        "duration": float(row.get("持続時間(s)", 0)),
                        "delay": float(row.get("遅延時間(s)", 0)),
                        "elapsed_time": float(row.get("経過時間(s) (セッション開始からの時間)", 0))
                    }
                    self.analysis_data.append(data_entry)
                
                # データ表示を更新
                self.update_data_display()
                
                messagebox.showinfo("成功", f"CSVファイルを読み込みました: {len(self.analysis_data)}件のデータ")
                
            except Exception as e:
                messagebox.showerror("エラー", f"CSVファイルの読み込みに失敗しました: {e}")
                print(f"CSV読み込みエラー: {e}")

    def update_data_display(self):
        """データ表示を更新"""
        self.data_display.config(state='normal')
        self.data_display.delete('1.0', tk.END)
        
        if not self.analysis_data:
            self.data_display.insert(tk.END, "データがありません。CSVファイルを読み込んでください。\n")
        else:
            # ヘッダー
            header_text = f" {'キー情報':<15} | {'持続時間(s)':<15} | {'遅延時間(s)':<15} | {'経過時間(s)':<15}\n"
            header_line = "=" * 80 + "\n"
            self.data_display.insert(tk.END, header_text)
            self.data_display.insert(tk.END, header_line)
            
            # データ行
            for entry in self.analysis_data:
                data_line = (
                    f" {entry['key_info']:<15} | "
                    f"{entry['duration']:.3f}{'':<12} | "
                    f"{entry['delay']:.3f}{'':<12} | "
                    f"{entry['elapsed_time']:.3f}{'':<12}\n"
                )
                self.data_display.insert(tk.END, data_line)
        
        self.data_display.config(state='disabled')

    def extract_features_from_keylog(self, log_data: pd.DataFrame, time_window_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """キーログデータから特徴量を抽出"""
        try:
            # データフレームの準備
            df = pd.DataFrame(log_data)
            
            if len(df) == 0:
                return np.array([]), np.array([])
            
            # 列名の統一
            if 'elapsed_time' not in df.columns:
                df['elapsed_time'] = df.get('経過時間(s) (セッション開始からの時間)', 0)
            if 'duration' not in df.columns:
                df['duration'] = df.get('持続時間(s)', 0)
            if 'delay' not in df.columns:
                df['delay'] = df.get('遅延時間(s)', 0)
            
            df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0)
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
            df['delay'] = pd.to_numeric(df['delay'], errors='coerce').fillna(0)

            max_time = df['elapsed_time'].max()
            segments = []
            
            start_time = 0.0
            while start_time < max_time + time_window_sec:
                end_time = start_time + time_window_sec
                window_data = df[(df['elapsed_time'] >= start_time) & (df['elapsed_time'] < end_time)]
                
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

        except Exception as e:
            print(f"特徴量抽出エラー: {e}")
            return np.array([]), np.array([])

    def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
        """観測データから感情ラベルの時系列を推定"""
        if not self.hmm_model or not self.state_names:
            return []
        try:
            logprob, state_sequence = self.hmm_model.decode(X, algorithm="viterbi")
            return [self.state_names[state] for state in state_sequence]
        except Exception as e:
            print(f"感情推定エラー: {e}")
            return []

    def analyze_emotions(self):
        """感情分析を実行"""
        if not self.analysis_data:
            messagebox.showwarning("警告", "分析データがありません。CSVファイルを読み込んでください。")
            return
            
        if not self.hmm_model:
            messagebox.showerror("エラー", "HMMモデルが読み込まれていません。")
            return
            
        try:
            # データから特徴量を抽出
            X, _ = self.extract_features_from_keylog(self.analysis_data)
            
            if X.size == 0:
                messagebox.showerror("エラー", "特徴量の抽出に失敗しました。")
                return
                
            # 感情推定
            estimated_emotions = self.predict_emotion_sequence(X)
            
            if not estimated_emotions:
                messagebox.showerror("エラー", "感情推定に失敗しました。")
                return
            
            # 結果を表示
            self.update_emotion_display("=== 感情分析結果 ===\n")
            for i, emotion in enumerate(estimated_emotions):
                time_start = i * 2
                time_end = (i + 1) * 2
                self.update_emotion_display(f"時間窓 {time_start}-{time_end}秒: {emotion}\n")
                
            # 統計情報を計算・表示
            self.calculate_and_display_statistics(estimated_emotions)
                
        except Exception as e:
            messagebox.showerror("エラー", f"感情分析中にエラーが発生しました: {e}")
            print(f"感情分析エラー: {e}")

    def calculate_and_display_statistics(self, emotions):
        """統計情報を計算して表示"""
        try:
            # 感情分布
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 統計情報を表示
            self.update_stats_display("=== 感情分布統計 ===\n")
            total_emotions = len(emotions)
            for emotion, count in emotion_counts.items():
                percentage = (count / total_emotions) * 100
                self.update_stats_display(f"{emotion}: {count}回 ({percentage:.1f}%)\n")
            
            # 最も頻繁な感情
            most_common = max(emotion_counts.items(), key=lambda x: x[1])
            self.update_stats_display(f"\n最も頻繁な感情: {most_common[0]} ({most_common[1]}回)\n")
            
            # 感情の変化回数
            changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            self.update_stats_display(f"感情変化回数: {changes}回\n")
            
            # 現在の感情を更新
            self.current_emotion = emotions[-1] if emotions else "未分析"
            self.current_emotion_label.config(text=self.current_emotion)
            
        except Exception as e:
            print(f"統計計算エラー: {e}")

    def update_emotion_display(self, text):
        """感情分析結果表示を更新"""
        self.emotion_display.config(state='normal')
        self.emotion_display.insert(tk.END, text)
        self.emotion_display.see(tk.END)
        self.emotion_display.config(state='disabled')

    def update_stats_display(self, text):
        """統計情報表示を更新"""
        self.stats_display.config(state='normal')
        self.stats_display.insert(tk.END, text)
        self.stats_display.see(tk.END)
        self.stats_display.config(state='disabled')

    def export_results(self):
        """分析結果をCSVファイルに出力"""
        if not self.analysis_data:
            messagebox.showwarning("警告", "分析データがありません。")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="分析結果をCSVで保存",
            initialfile=CSV_FILENAME
        )
        
        if not filename:
            return 

        try:
            # 元データと分析結果を結合
            df = pd.DataFrame(self.analysis_data)
            
            # 感情分析を実行して結果を追加
            if self.hmm_model and len(self.analysis_data) > 0:
                X, _ = self.extract_features_from_keylog(self.analysis_data)
                if X.size > 0:
                    estimated_emotions = self.predict_emotion_sequence(X)
                    # 時間窓ごとの感情を追加
                    df['推定感情'] = estimated_emotions[:len(df)]
            
            df.to_csv(filename, index=False, encoding='utf-8')
            messagebox.showinfo("成功", f"分析結果を {filename} に保存しました。")
            
        except Exception as e:
            messagebox.showerror("エラー", f"CSV出力中にエラーが発生しました: {e}")

    def clear_display(self):
        """表示をクリア"""
        self.analysis_data = []
        self.current_emotion = "未分析"
        self.current_emotion_label.config(text=self.current_emotion)

        # 各表示をクリア
        for display in [self.data_display, self.emotion_display, self.stats_display]:
            display.config(state='normal')
            display.delete('1.0', tk.END)
            display.config(state='disabled')

    def on_closing(self):
        """ウィンドウが閉じられる時の処理"""
        self.master.destroy()

# --- メイン処理 ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EmotionAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"アプリケーションの起動に失敗しました: {e}")
        import traceback
        traceback.print_exc()
