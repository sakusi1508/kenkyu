import time
import csv
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog
from pynput import keyboard

# CSV出力ファイル名 (定数として保持)
CSV_FILENAME = "mizuki2025output.csv"

# --- Tkinterアプリケーションクラス ---
class KeyloggerApp:
    def __init__(self, master):
        self.master = master
        master.title("Python キーロガー (GUI)")
        master.geometry("750x500")

        # 状態変数 (すべてインスタンス変数で管理)
        self.log_data = [] # 記録された全ログデータ
        self.last_release_time = None # 遅延時間計算のための最後の解放時刻
        self.listener = None 
        self.keylogger_running = False
        
        # キーリピート対策: 現在押下されているキーを追跡
        self.pressed_keys = set() 
        
        # 経過時間の基準時間 (リセット可能)
        self.program_start_time = time.time()
        self.session_start_time = self.program_start_time 

        # ログ表示エリア (ScrolledText)
        self.log_display = scrolledtext.ScrolledText(master, state='disabled', wrap='word', width=90, height=20, font=("Courier", 10))
        self.log_display.pack(pady=10, padx=10)
        
        # ヘッダーを最初に表示
        self.update_display(header=True)

        # ボタンフレーム
        button_frame = tk.Frame(master)
        button_frame.pack(pady=5)

        # 開始/停止ボタン
        self.start_stop_button = tk.Button(button_frame, text="ロガー開始", command=self.toggle_keylogger, width=15, bg='lightgreen')
        self.start_stop_button.pack(side=tk.LEFT, padx=10)

        # CSV保存ボタン
        self.export_button = tk.Button(button_frame, text="CSVへ保存", command=self.export_to_csv, width=15)
        self.export_button.pack(side=tk.LEFT, padx=10)

        # クリアボタン (タイマーリセット機能を内包)
        self.clear_button = tk.Button(button_frame, text="表示クリア＆タイマーリセット", command=self.clear_display, width=25)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        # 終了時のイベントハンドラを設定
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- キーロガーロジック ---
    def _get_key_info(self, key):
        """キーオブジェクトから表示用のキー情報を取得"""
        try:
            return key.char
        except AttributeError:
            return str(key).replace('Key.', '')

    def on_press(self, key):
        """キーが押された時の処理"""
        key_info = self._get_key_info(key)
        
        # 【修正点】キーリピート対策: 既に押されているキーは無視
        if key_info in self.pressed_keys:
            return
        
        # 初回押下であればセットに追加
        self.pressed_keys.add(key_info)

        press_time = time.time()
        
        # 遅延時間を計算
        delay = 0.0
        if self.last_release_time is not None:
            delay = press_time - self.last_release_time
        
        # ログエントリを作成 (session_start_timeを基準に経過時間を計算)
        log_entry = {
            "key_info": key_info,
            "press_time": press_time,
            "release_time": None,
            "duration": 0.0,
            "delay": delay,
            "elapsed_time": press_time - self.session_start_time 
        }
        self.log_data.append(log_entry)

    def on_release(self, key):
        """キーが離された時の処理"""
        release_time = time.time()
        current_key_info = self._get_key_info(key)
            
        # 【修正点】キーリピート対策: 追跡セットから削除
        if current_key_info in self.pressed_keys:
            self.pressed_keys.remove(current_key_info)
            
        # ログエントリを検索し、持続時間を計算して更新
        for entry in reversed(self.log_data):
            # 該当キーで、まだ解放時刻が記録されていないものを探す
            if entry["key_info"] == current_key_info and entry["release_time"] is None:
                # 持続時間を計算
                duration = release_time - entry["press_time"]
                
                # エントリを更新
                entry["release_time"] = release_time
                entry["duration"] = duration
                
                self.update_display(entry)
                break
                
        # 次のキーを押す時点での遅延時間計算のために、現在の解放時刻を記録
        self.last_release_time = release_time

    # --- GUI制御ロジック ---
    def update_display(self, entry=None, header=False):
        """GUIのテキストエリアにログを追記する"""
        self.log_display.config(state='normal') 
        
        if header:
            # ヘッダー行を整形
            header_text = f" {'キー情報':<15} | {'持続時間(s)':<15} | {'遅延時間(s)':<15} | {'経過時間(s) (セッション)':<15}\n"
            header_line = "=" * 80 + "\n"
            self.log_display.insert(tk.END, header_text)
            self.log_display.insert(tk.END, header_line)
        
        if entry and entry["release_time"] is not None:
            # ログデータ行を整形
            log_line = (
                f" {entry['key_info']:<15} | "
                f"{entry['duration']:.3f}{'':<12} | "
                f"{entry['delay']:.3f}{'':<12} | "
                f"{entry['elapsed_time']:.3f}{'':<12}\n"
            )
            self.log_display.insert(tk.END, log_line)
            self.log_display.see(tk.END) # 一番下までスクロール
            
        self.log_display.config(state='disabled') 

    def clear_display(self):
        """表示エリアのログと、経過時間の基準時間をクリアする"""
        
        # 【修正点】タイマーリセット: 経過時間の基準時間を現在時刻にリセット
        self.session_start_time = time.time() 
        self.last_release_time = None # 遅延時間計算のための基準もリセット
        self.pressed_keys.clear() # 現在押されているキーの状態もリセット

        self.log_display.config(state='normal')
        self.log_display.delete('1.0', tk.END)
        self.update_display(header=True) 
        self.log_display.config(state='disabled')

    def start_keylogger_thread(self):
        """キーロガーリスナーを別スレッドで起動する"""
        if not self.keylogger_running:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            self.keylogger_running = True
            self.start_stop_button.config(text="ロガー停止", bg='lightcoral')
            print("キーロガーが開始されました。")

    def stop_keylogger_thread(self):
        """キーロガーリスナーを停止する"""
        if self.keylogger_running and self.listener:
            self.listener.stop()
            self.listener = None
            self.keylogger_running = False
            self.start_stop_button.config(text="ロガー開始", bg='lightgreen')
            print("キーロガーが停止されました。")

    def toggle_keylogger(self):
        """開始/停止を切り替える"""
        if self.keylogger_running:
            self.stop_keylogger_thread()
        else:
            self.start_keylogger_thread()

    def export_to_csv(self):
        """記録されたログデータをCSVファイルに出力する"""
        if not self.log_data:
            print("ログデータがありません。")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="ログデータをCSVで保存",
            initialfile=CSV_FILENAME
        )
        
        if not filename:
            return 

        # CSVのヘッダー定義
        fieldnames = [
            "キー情報",
            "持続時間(s)",
            "遅延時間(s)",
            "経過時間(s) (セッション開始からの時間)"
        ]

        csv_rows = []
        for data in self.log_data:
            # 解放されたキーのみを記録対象とする
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
                
            print(f"✅ ログデータを正常に {filename} に出力しました。")
        except Exception as e:
            print(f"❌ CSV出力中にエラーが発生しました: {e}")

    def on_closing(self):
        """ウィンドウが閉じられる時の処理"""
        self.stop_keylogger_thread()
        self.master.destroy()


# --- メイン処理 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = KeyloggerApp(root)
    # アプリケーション起動時に自動でキーロガーを開始
    app.start_keylogger_thread() 
    root.mainloop()