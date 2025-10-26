import time
import csv
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
import joblib

# pynputの代わりに、より安全なアプローチを使用
try:
    from pynput import keyboard
    KEYLOGGER_AVAILABLE = True
except ImportError:
    KEYLOGGER_AVAILABLE = False
    print("⚠️ pynputが利用できません。キーロガー機能は無効になります。")

# --- 定数定義 ---
CSV_FILENAME = "mizuki2025output.csv"
EMO_CSV_FILENAME = "EMOoutput.csv"
MODEL_FILENAME = 'hmm_emotion_model_4states.pkl'
STATES_FILENAME = 'hmm_emotion_model_4states_states.pkl'

# HMMの隠れ状態（感情ラベル）：4つ
EMOTIONAL_STATES = [
    '感覚運動的興奮', 
    '難解・頭脳型', 
    '和みと癒し', 
    '設定状況の魅力' 
]
N_STATES = len(EMOTIONAL_STATES)
N_FEATURES = 5  # 特徴量の次元数

class SafeEmotionKeyloggerApp:
    def __init__(self, master):
        self.master = master
        master.title("感情分析キーロガーアプリ (安全版)")
        master.geometry("1000x700")
        
        # 状態変数
        self.log_data = []
        self.last_release_time = None
        self.listener = None 
        self.keylogger_running = False
        self.pressed_keys = set()
        self.program_start_time = time.time()
        self.session_start_time = self.program_start_time
        
        # HMMモデル関連
        self.hmm_model = None
        self.emotion_history = []
        self.current_emotion = "未分析"
        
        # キーロガー利用可能性チェック
        self.keylogger_enabled = KEYLOGGER_AVAILABLE
        
        # GUIの構築
        self.create_widgets()
        
        # HMMモデルの読み込み試行
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
        title_label = tk.Label(control_frame, text="感情分析キーロガー (安全版)", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # キーロガー状態表示
        status_frame = tk.Frame(control_frame)
        status_frame.pack()
        
        if self.keylogger_enabled:
            status_text = "✅ キーロガー機能が利用可能です"
            status_color = "green"
        else:
            status_text = "⚠️ キーロガー機能が無効です (pynputが利用できません)"
            status_color = "orange"
            
        status_label = tk.Label(status_frame, text=status_text, fg=status_color, font=("Arial", 10))
        status_label.pack()
        
        # ボタンフレーム
        button_frame = tk.Frame(control_frame)
        button_frame.pack()
        
        # キーロガー制御ボタン
        if self.keylogger_enabled:
            self.start_stop_button = tk.Button(button_frame, text="キーロガー開始", 
                                              command=self.toggle_keylogger, 
                                              width=15, bg='lightgreen')
        else:
            self.start_stop_button = tk.Button(button_frame, text="キーロガー無効", 
                                              command=self.show_keylogger_error, 
                                              width=15, bg='lightgray', state='disabled')
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 手動データ入力ボタン
        self.manual_input_button = tk.Button(button_frame, text="手動データ入力", 
                                           command=self.manual_data_input, 
                                           width=15, bg='lightblue')
        self.manual_input_button.pack(side=tk.LEFT, padx=5)
        
        # 感情分析ボタン
        self.analyze_button = tk.Button(button_frame, text="感情分析実行", 
                                       command=self.analyze_emotions, 
                                       width=15, bg='lightblue')
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        # CSV保存ボタン
        self.export_button = tk.Button(button_frame, text="CSV保存", 
                                     command=self.export_to_csv, 
                                     width=15)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # CSV読み込みボタン
        self.import_button = tk.Button(button_frame, text="CSV読み込み", 
                                     command=self.import_csv, 
                                     width=15)
        self.import_button.pack(side=tk.LEFT, padx=5)
        
        # クリアボタン
        self.clear_button = tk.Button(button_frame, text="クリア", 
                                     command=self.clear_display, 
                                     width=15)
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
        
        # キーログ表示タブ
        self.log_frame = tk.Frame(notebook)
        notebook.add(self.log_frame, text="キーログ")
        
        self.log_display = scrolledtext.ScrolledText(self.log_frame, state='disabled', 
                                                   wrap='word', width=100, height=20, 
                                                   font=("Courier", 10))
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 感情分析結果タブ
        self.emotion_frame = tk.Frame(notebook)
        notebook.add(self.emotion_frame, text="感情分析結果")
        
        self.emotion_display = scrolledtext.ScrolledText(self.emotion_frame, state='disabled', 
                                                       wrap='word', width=100, height=20, 
                                                       font=("Courier", 10))
        self.emotion_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # リアルタイム分析タブ
        self.realtime_frame = tk.Frame(notebook)
        notebook.add(self.realtime_frame, text="リアルタイム分析")
        
        self.realtime_display = scrolledtext.ScrolledText(self.realtime_frame, state='disabled', 
                                                         wrap='word', width=100, height=20, 
                                                         font=("Courier", 10))
        self.realtime_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ヘッダーを最初に表示
        self.update_display(header=True)

    def show_keylogger_error(self):
        """キーロガー無効時のメッセージ表示"""
        messagebox.showinfo("キーロガー無効", 
                           "キーロガー機能が無効です。\n\n"
                           "手動データ入力またはCSV読み込み機能をご利用ください。\n\n"
                           "キーロガーを有効にするには:\n"
                           "1. pynputライブラリをインストール\n"
                           "2. macOSのアクセシビリティ権限を設定")

    def manual_data_input(self):
        """手動データ入力ダイアログ"""
        dialog = tk.Toplevel(self.master)
        dialog.title("手動データ入力")
        dialog.geometry("400x300")
        dialog.transient(self.master)
        dialog.grab_set()
        
        # 入力フィールド
        tk.Label(dialog, text="キー情報:").pack(pady=5)
        key_entry = tk.Entry(dialog, width=20)
        key_entry.pack(pady=5)
        
        tk.Label(dialog, text="持続時間(s):").pack(pady=5)
        duration_entry = tk.Entry(dialog, width=20)
        duration_entry.pack(pady=5)
        
        tk.Label(dialog, text="遅延時間(s):").pack(pady=5)
        delay_entry = tk.Entry(dialog, width=20)
        delay_entry.pack(pady=5)
        
        def add_data():
            try:
                key_info = key_entry.get()
                duration = float(duration_entry.get())
                delay = float(delay_entry.get())
                
                if key_info:
                    current_time = time.time()
                    log_entry = {
                        "key_info": key_info,
                        "press_time": current_time,
                        "release_time": current_time + duration,
                        "duration": duration,
                        "delay": delay,
                        "elapsed_time": current_time - self.session_start_time
                    }
                    self.log_data.append(log_entry)
                    self.update_display(log_entry)
                    dialog.destroy()
                else:
                    messagebox.showerror("エラー", "キー情報を入力してください。")
            except ValueError:
                messagebox.showerror("エラー", "数値を正しく入力してください。")
        
        tk.Button(dialog, text="追加", command=add_data).pack(pady=10)
        tk.Button(dialog, text="キャンセル", command=dialog.destroy).pack(pady=5)

    def import_csv(self):
        """CSVファイルを読み込み"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="CSVファイルを選択"
        )
        
        if filename:
            try:
                df = pd.read_csv(filename)
                self.log_data = []
                
                for _, row in df.iterrows():
                    log_entry = {
                        "key_info": row.get("キー情報", ""),
                        "press_time": 0,  # ダミー値
                        "release_time": 0,  # ダミー値
                        "duration": float(row.get("持続時間(s)", 0)),
                        "delay": float(row.get("遅延時間(s)", 0)),
                        "elapsed_time": float(row.get("経過時間(s) (セッション開始からの時間)", 0))
                    }
                    self.log_data.append(log_entry)
                
                # 表示を更新
                self.clear_display()
                for entry in self.log_data:
                    if entry["release_time"] is not None:
                        self.update_display(entry)
                
                messagebox.showinfo("成功", f"CSVファイルを読み込みました: {len(self.log_data)}件のデータ")
                
            except Exception as e:
                messagebox.showerror("エラー", f"CSVファイルの読み込みに失敗しました: {e}")

    def load_hmm_model(self):
        """HMMモデルを読み込む"""
        try:
            if os.path.exists(MODEL_FILENAME) and os.path.exists(STATES_FILENAME):
                self.hmm_model = joblib.load(MODEL_FILENAME)
                self.state_names = joblib.load(STATES_FILENAME)
                self.update_emotion_display("✅ HMMモデルが正常に読み込まれました。")
            else:
                self.update_emotion_display("⚠️ HMMモデルが見つかりません。先にモデルを訓練してください。")
        except Exception as e:
            self.update_emotion_display(f"❌ HMMモデルの読み込みエラー: {e}")

    def _get_key_info(self, key):
        """キーオブジェクトから表示用のキー情報を取得"""
        try:
            return key.char
        except AttributeError:
            return str(key).replace('Key.', '')

    def on_press(self, key):
        """キーが押された時の処理"""
        try:
            key_info = self._get_key_info(key)
            
            if key_info in self.pressed_keys:
                return
            
            self.pressed_keys.add(key_info)
            press_time = time.time()
            
            delay = 0.0
            if self.last_release_time is not None:
                delay = press_time - self.last_release_time
            
            log_entry = {
                "key_info": key_info,
                "press_time": press_time,
                "release_time": None,
                "duration": 0.0,
                "delay": delay,
                "elapsed_time": press_time - self.session_start_time 
            }
            self.log_data.append(log_entry)
        except Exception as e:
            print(f"キー押下処理エラー: {e}")

    def on_release(self, key):
        """キーが離された時の処理"""
        try:
            release_time = time.time()
            current_key_info = self._get_key_info(key)
                
            if current_key_info in self.pressed_keys:
                self.pressed_keys.remove(current_key_info)
                
            for entry in reversed(self.log_data):
                if entry["key_info"] == current_key_info and entry["release_time"] is None:
                    duration = release_time - entry["press_time"]
                    entry["release_time"] = release_time
                    entry["duration"] = duration
                    
                    self.update_display(entry)
                    # リアルタイム分析を実行
                    self.realtime_analysis()
                    break
                    
            self.last_release_time = release_time
        except Exception as e:
            print(f"キー解放処理エラー: {e}")

    def update_display(self, entry=None, header=False):
        """キーログ表示を更新"""
        self.log_display.config(state='normal') 
        
        if header:
            header_text = f" {'キー情報':<15} | {'持続時間(s)':<15} | {'遅延時間(s)':<15} | {'経過時間(s)':<15}\n"
            header_line = "=" * 80 + "\n"
            self.log_display.insert(tk.END, header_text)
            self.log_display.insert(tk.END, header_line)
        
        if entry and entry["release_time"] is not None:
            log_line = (
                f" {entry['key_info']:<15} | "
                f"{entry['duration']:.3f}{'':<12} | "
                f"{entry['delay']:.3f}{'':<12} | "
                f"{entry['elapsed_time']:.3f}{'':<12}\n"
            )
            self.log_display.insert(tk.END, log_line)
            self.log_display.see(tk.END)
            
        self.log_display.config(state='disabled')

    def realtime_analysis(self):
        """リアルタイム感情分析"""
        try:
            if not self.hmm_model or len(self.log_data) < 5:
                return
                
            # 最近のデータから特徴量を抽出
            recent_data = self.log_data[-50:]  # 最近50個のキーイベント
            df = pd.DataFrame(recent_data)
            
            if len(df) < 3:
                return
                
            # 特徴量抽出
            X, _ = self.extract_features_from_keylog(df)
            
            if X.size > 0:
                # 感情推定
                estimated_emotions = self.predict_emotion_sequence(X)
                if estimated_emotions:
                    self.current_emotion = estimated_emotions[-1]
                    self.current_emotion_label.config(text=self.current_emotion)
                    
                    # リアルタイム表示を更新
                    self.update_realtime_display(estimated_emotions[-1])
                    
        except Exception as e:
            print(f"リアルタイム分析エラー: {e}")

    def extract_features_from_keylog(self, log_data: pd.DataFrame, time_window_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """キーログデータから特徴量を抽出"""
        try:
            # 列名の統一
            if '経過時間(s) (セッション開始からの時間)' in log_data.columns:
                log_data.rename(columns={'経過時間(s) (セッション開始からの時間)': 'elapsed_time'}, inplace=True)
            if '持続時間(s)' in log_data.columns:
                log_data.rename(columns={'持続時間(s)': 'duration'}, inplace=True)
            if '遅延時間(s)' in log_data.columns:
                log_data.rename(columns={'遅延時間(s)': 'delay'}, inplace=True)
            
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
                if np.isnan(duration_var): duration_var = 0.0
                total_active_duration = window_data['duration'].sum()
                stop_time_ratio = (time_window_sec - total_active_duration) / time_window_sec
                stop_time_ratio = max(0.0, min(1.0, stop_time_ratio))
                
                feature_vector = [duration_mean, delay_mean, apm, duration_var, stop_time_ratio]
                
            segments.append(feature_vector)
            start_time = end_time

        X = np.array(segments)
        lengths = np.array([len(X)])
        
        return X, lengths

    def predict_emotion_sequence(self, X: np.ndarray) -> List[str]:
        """観測データから感情ラベルの時系列を推定"""
        if not self.hmm_model:
            return []
        try:
            logprob, state_sequence = self.hmm_model.decode(X, algorithm="viterbi")
            return [self.state_names[state] for state in state_sequence]
        except Exception as e:
            print(f"感情推定エラー: {e}")
            return []

    def update_realtime_display(self, emotion):
        """リアルタイム分析結果を表示"""
        self.realtime_display.config(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        self.realtime_display.insert(tk.END, f"[{timestamp}] 推定感情: {emotion}\n")
        self.realtime_display.see(tk.END)
        self.realtime_display.config(state='disabled')

    def analyze_emotions(self):
        """感情分析を実行"""
        if not self.log_data:
            messagebox.showwarning("警告", "キーログデータがありません。")
            return
            
        if not self.hmm_model:
            messagebox.showerror("エラー", "HMMモデルが読み込まれていません。")
            return
            
        try:
            # 全データから特徴量を抽出
            df = pd.DataFrame(self.log_data)
            X, _ = self.extract_features_from_keylog(df)
            
            if X.size == 0:
                messagebox.showerror("エラー", "特徴量の抽出に失敗しました。")
                return
                
            # 感情推定
            estimated_emotions = self.predict_emotion_sequence(X)
            
            # 結果を表示
            self.update_emotion_display("=== 感情分析結果 ===\n")
            for i, emotion in enumerate(estimated_emotions):
                time_start = i * 2
                time_end = (i + 1) * 2
                self.update_emotion_display(f"時間窓 {time_start}-{time_end}秒: {emotion}\n")
                
            # 統計情報
            emotion_counts = {}
            for emotion in estimated_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
            self.update_emotion_display("\n=== 感情分布 ===\n")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(estimated_emotions)) * 100
                self.update_emotion_display(f"{emotion}: {count}回 ({percentage:.1f}%)\n")
                
        except Exception as e:
            messagebox.showerror("エラー", f"感情分析中にエラーが発生しました: {e}")

    def update_emotion_display(self, text):
        """感情分析結果表示を更新"""
        self.emotion_display.config(state='normal')
        self.emotion_display.insert(tk.END, text)
        self.emotion_display.see(tk.END)
        self.emotion_display.config(state='disabled')

    def clear_display(self):
        """表示をクリア"""
        self.session_start_time = time.time() 
        self.last_release_time = None
        self.pressed_keys.clear()
        self.current_emotion = "未分析"
        self.current_emotion_label.config(text=self.current_emotion)

        # ログ表示をクリア
        self.log_display.config(state='normal')
        self.log_display.delete('1.0', tk.END)
        self.update_display(header=True) 
        self.log_display.config(state='disabled')
        
        # 感情分析結果をクリア
        self.emotion_display.config(state='normal')
        self.emotion_display.delete('1.0', tk.END)
        self.emotion_display.config(state='disabled')
        
        # リアルタイム分析結果をクリア
        self.realtime_display.config(state='normal')
        self.realtime_display.delete('1.0', tk.END)
        self.realtime_display.config(state='disabled')

    def start_keylogger_thread(self):
        """キーロガーリスナーを別スレッドで起動"""
        if not self.keylogger_running and self.keylogger_enabled:
            try:
                self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
                self.listener.start()
                self.keylogger_running = True
                self.start_stop_button.config(text="キーロガー停止", bg='lightcoral')
                print("✅ キーロガーが開始されました。")
            except Exception as e:
                print(f"❌ キーロガーの開始に失敗しました: {e}")
                messagebox.showerror("エラー", f"キーロガーの開始に失敗しました:\n{e}\n\n権限の問題の可能性があります。")
                return

    def stop_keylogger_thread(self):
        """キーロガーリスナーを停止"""
        if self.keylogger_running and self.listener:
            self.listener.stop()
            self.listener = None
            self.keylogger_running = False
            self.start_stop_button.config(text="キーロガー開始", bg='lightgreen')

    def toggle_keylogger(self):
        """開始/停止を切り替える"""
        if self.keylogger_running:
            self.stop_keylogger_thread()
        else:
            self.start_keylogger_thread()

    def export_to_csv(self):
        """記録されたログデータをCSVファイルに出力"""
        if not self.log_data:
            messagebox.showwarning("警告", "ログデータがありません。")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="ログデータをCSVで保存",
            initialfile=CSV_FILENAME
        )
        
        if not filename:
            return 

        fieldnames = [
            "キー情報",
            "持続時間(s)",
            "遅延時間(s)",
            "経過時間(s) (セッション開始からの時間)"
        ]

        csv_rows = []
        for data in self.log_data:
            if data["release_time"] is not None:
                csv_rows.append({
                    "キー情報": data["key_info"],
                    "持続時間(s)": f"{data['duration']:.3f}",
                    "遅延時間(s)": f"{data['delay']:.3f}",
                    "経過時間(s) (セッション開始からの時間)": f"{data['elapsed_time']:.3f}"
                })

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
                
            messagebox.showinfo("成功", f"ログデータを {filename} に保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"CSV出力中にエラーが発生しました: {e}")

    def on_closing(self):
        """ウィンドウが閉じられる時の処理"""
        if self.keylogger_running:
            self.stop_keylogger_thread()
        self.master.destroy()

# --- メイン処理 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SafeEmotionKeyloggerApp(root)
    root.mainloop()


