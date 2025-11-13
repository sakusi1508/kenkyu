"""
S.M.A.R.T. ゲームレコメンデーションシステム - GUIアプリケーション

14個の感情ラベルを5個の型に変換し、4個の主観感情との共通印象を比較し、
キーログから4個の主観感情を推定し、5個の型から最も適合するゲームタイトルをレコメンドする
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ローカルモジュールをインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_converter import (
    convert_to_emotional_labels, convert_to_game_types,
    compare_emotional_and_type_similarity, FACTOR_LIST
)
from keylog_analyzer import estimate_emotions_from_keylog
from game_recommender import recommend_games


class SmartGameRecommendationGUI:
    """S.M.A.R.T. ゲームレコメンデーションGUIアプリケーション"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("S.M.A.R.T. ゲームレコメンデーションシステム")
        self.root.geometry("1000x800")
        
        # 変数の初期化
        self.keylog_path = tk.StringVar()
        self.hmm_model_path = tk.StringVar()
        self.factor_csv_path = tk.StringVar()
        self.game_scores_path = tk.StringVar()
        self.time_window = tk.DoubleVar(value=30.0)
        self.top_n = tk.IntVar(value=5)
        self.output_path = tk.StringVar()
        
        # デフォルトパスの設定
        default_model_path = os.path.join(PROJECT_ROOT, 'HMMleran', 'hmm_emotion_model_4states.pkl')
        if os.path.exists(default_model_path):
            self.hmm_model_path.set(default_model_path)
        
        default_game_scores_path = os.path.join(PROJECT_ROOT, 'syuyougametitle_score', 'merged_game_scores.csv')
        if os.path.exists(default_game_scores_path):
            self.game_scores_path.set(default_game_scores_path)
        
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
        title_label = ttk.Label(main_frame, text="S.M.A.R.T. ゲームレコメンデーションシステム", 
                               font=("Helvetica", 16, "bold"))
        title_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # 説明
        desc_label = ttk.Label(main_frame, 
                               text="14個の感情ラベルを5個の型に変換し、キーログから4個の主観感情を推定してゲームをレコメンドします",
                               font=("Helvetica", 10))
        desc_label.grid(row=row, column=0, columnspan=3, pady=5)
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
        
        # キーログCSV
        ttk.Label(input_frame, text="キーログCSV:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.keylog_path, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.keylog_path, "キーログCSVファイルを選択", 
                                                  [("CSV files", "*.csv")])).grid(
            row=0, column=2, pady=5)
        
        # 14因子CSV（オプション）
        ttk.Label(input_frame, text="14因子CSV (オプション):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.factor_csv_path, width=60).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.factor_csv_path, "14因子CSVファイルを選択", 
                                                  [("CSV files", "*.csv")])).grid(
            row=1, column=2, pady=5)
        
        # HMMモデルファイル（オプション）
        ttk.Label(input_frame, text="HMMモデルファイル (オプション):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.hmm_model_path, width=60).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.hmm_model_path, "HMMモデルファイルを選択", 
                                                  [("PKL files", "*.pkl")])).grid(
            row=2, column=2, pady=5)
        
        # ゲームスコアCSV
        ttk.Label(input_frame, text="ゲームスコアCSV:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.game_scores_path, width=60).grid(
            row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(input_frame, text="参照", 
                  command=lambda: self.select_file(self.game_scores_path, "ゲームスコアCSVファイルを選択", 
                                                  [("CSV files", "*.csv")])).grid(
            row=3, column=2, pady=5)
        
        # パラメータ設定セクション
        param_frame = ttk.LabelFrame(main_frame, text="パラメータ設定", padding="10")
        param_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        param_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 時間窓サイズ
        ttk.Label(param_frame, text="時間窓サイズ (秒):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(param_frame, from_=10.0, to=300.0, increment=10.0, 
                   textvariable=self.time_window, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # トップN
        ttk.Label(param_frame, text="レコメンド数 (Top N):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(param_frame, from_=1, to=20, textvariable=self.top_n, width=20).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 出力ファイル
        ttk.Label(param_frame, text="出力CSV (オプション):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(param_frame, textvariable=self.output_path, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(param_frame, text="参照", 
                  command=lambda: self.select_file(self.output_path, "出力CSVファイルを選択", 
                                                  [("CSV files", "*.csv")], save=True)).grid(
            row=2, column=2, pady=5)
        
        # ボタンセクション
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        ttk.Button(button_frame, text="レコメンデーション実行", 
                  command=self.execute_recommendation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="クリア", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # 結果表示セクション
        result_frame = ttk.LabelFrame(main_frame, text="結果", padding="10")
        result_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        # 結果テキストエリア
        self.result_text = scrolledtext.ScrolledText(result_frame, width=100, height=25, wrap=tk.WORD)
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
        self.keylog_path.set("")
        self.factor_csv_path.set("")
        self.output_path.set("")
        self.result_text.delete(1.0, tk.END)
        messagebox.showinfo("クリア", "すべての入力と結果をクリアしました。")
    
    def execute_recommendation(self):
        """レコメンデーションを実行"""
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "レコメンデーション処理を開始します...\n\n")
            self.root.update()
            
            # 入力の検証
            keylog = self.keylog_path.get().strip()
            factor_csv = self.factor_csv_path.get().strip()
            
            if not keylog and not factor_csv:
                messagebox.showerror("エラー", 
                    "キーログCSVまたは14因子CSVのいずれかを指定してください。")
                return
            
            # 14因子の取得
            factors = None
            emotional_similarities = None
            game_type_percentages = None
            common_impression_scores = None
            recommended_games = []
            emotion_distribution = None
            emotion_percentages = None
            dominant_emotion = None
            
            if factor_csv:
                # 14因子CSVから読み込み
                if not os.path.exists(factor_csv):
                    messagebox.showerror("エラー", f"ファイルが見つかりません: {factor_csv}")
                    return
                
                self.result_text.insert(tk.END, f"14因子CSVを読み込み中: {factor_csv}\n")
                self.root.update()
                
                df_factors = pd.read_csv(factor_csv)
                # 最初の行を使用（複数行の場合は最初の行）
                if 'game_title' in df_factors.columns:
                    factors = {factor: df_factors.iloc[0].get(factor, 0.0) for factor in FACTOR_LIST}
                else:
                    factors = {factor: df_factors.iloc[0].get(factor, 0.0) for factor in FACTOR_LIST}
            
            elif keylog:
                # キーログから推定
                if not os.path.exists(keylog):
                    messagebox.showerror("エラー", f"ファイルが見つかりません: {keylog}")
                    return
                
                self.result_text.insert(tk.END, f"キーログを処理中: {keylog}\n")
                self.root.update()
                
                hmm_model = self.hmm_model_path.get().strip() if self.hmm_model_path.get().strip() else None
                
                result = estimate_emotions_from_keylog(
                    keylog_path=keylog,
                    hmm_model_path=hmm_model if hmm_model and os.path.exists(hmm_model) else None,
                    time_window_sec=self.time_window.get(),
                    top_n=self.top_n.get()
                )
                
                factors = result.get('factors', {})
                emotional_similarities = result.get('emotional_similarities', {})
                game_type_percentages = result.get('game_type_percentages', {})
                common_impression_scores = result.get('common_impression_scores', {})
                emotion_distribution = result.get('emotion_distribution')
                emotion_percentages = result.get('emotion_percentages')
                dominant_emotion = result.get('dominant_emotion')
                recommended_games = result.get('recommended_games', [])
            
            else:
                messagebox.showerror("エラー", 
                    "キーログCSVまたは14因子CSVのいずれかを指定してください。")
                return
            
            # 14因子CSVから直接変換する場合
            if factor_csv and factors and emotional_similarities is None:
                self.result_text.insert(tk.END, "14因子から5個の主観感情への変換中...\n")
                self.root.update()
                
                emotional_similarities = convert_to_emotional_labels(factors)
                
                self.result_text.insert(tk.END, "14因子から5個の型への変換中...\n")
                self.root.update()
                
                game_type_percentages = convert_to_game_types(factors)
                
                self.result_text.insert(tk.END, "共通印象を比較中...\n")
                self.root.update()
                
                common_impression_scores = compare_emotional_and_type_similarity(
                    emotional_similarities, game_type_percentages
                )
            
            # 結果の検証
            if emotional_similarities is None or game_type_percentages is None or common_impression_scores is None:
                messagebox.showerror("エラー", "感情・型の推定に失敗しました。")
                return
            
            if not recommended_games:
                # ゲームレコメンデーション（従来ロジックのフォールバック）
                self.result_text.insert(tk.END, "ゲームをレコメンド中...\n\n")
                self.root.update()
                
                game_scores_path = self.game_scores_path.get().strip() if self.game_scores_path.get().strip() else None
                recommended_games = recommend_games(
                    game_type_percentages=game_type_percentages,
                    common_impression_scores=common_impression_scores,
                    game_scores_path=game_scores_path,
                    top_n=self.top_n.get()
                )
            else:
                self.result_text.insert(tk.END, "HMMモデルに基づくレコメンド結果を使用します。\n\n")
                self.root.update()
            
            # 結果を表示
            self.display_result(
                factors if isinstance(factors, dict) else {},
                emotional_similarities,
                game_type_percentages,
                common_impression_scores,
                recommended_games,
                emotion_distribution=emotion_distribution,
                emotion_percentages=emotion_percentages,
                dominant_emotion=dominant_emotion
            )
            
            # 出力ファイルが指定されている場合
            if self.output_path.get():
                self.save_result_to_csv(
                    emotional_similarities,
                    game_type_percentages,
                    common_impression_scores,
                    recommended_games
                )
        
        except Exception as e:
            error_msg = f"レコメンデーション処理中にエラーが発生しました: {str(e)}"
            self.result_text.insert(tk.END, f"❌ {error_msg}\n")
            messagebox.showerror("エラー", error_msg)
            import traceback
            self.result_text.insert(tk.END, traceback.format_exc())
    
    def display_result(self, factors: Dict, emotional_similarities: Dict, 
                      game_type_percentages: Dict, common_impression_scores: Dict,
                      recommended_games: List, emotion_distribution: Optional[Dict[str, int]] = None,
                      emotion_percentages: Optional[Dict[str, float]] = None,
                      dominant_emotion: Optional[str] = None):
        """結果を表示"""
        self.result_text.insert(tk.END, "="*80 + "\n")
        self.result_text.insert(tk.END, "レコメンデーション結果\n")
        self.result_text.insert(tk.END, "="*80 + "\n\n")
        
        if dominant_emotion:
            self.result_text.insert(tk.END, f"主要感情: {dominant_emotion}\n\n")
        
        if emotion_distribution:
            self.result_text.insert(tk.END, "【感情の出現回数】\n")
            sorted_distribution = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_distribution:
                percentage = 0.0
                if emotion_percentages and emotion in emotion_percentages:
                    percentage = emotion_percentages[emotion] * 100
                self.result_text.insert(
                    tk.END,
                    f"  {emotion}: {count}回" + (f" ({percentage:.2f}%)" if percentage else "") + "\n"
                )
            self.result_text.insert(tk.END, "\n")
        
        # 14因子の表示
        if factors:
            self.result_text.insert(tk.END, "【14因子の値】\n")
            for factor in FACTOR_LIST:
                value = factors.get(factor, 0.0)
                self.result_text.insert(tk.END, f"  {factor}: {value:.3f}\n")
            self.result_text.insert(tk.END, "\n")
        
        # 主観感情への適合度（4個）
        self.result_text.insert(tk.END, "【4個の主観感情への適合度】\n")
        sorted_emotions = sorted(emotional_similarities.items(), key=lambda x: x[1], reverse=True)
        for emotion, similarity in sorted_emotions:
            self.result_text.insert(tk.END, f"  {emotion}: {similarity:.3f}\n")
        self.result_text.insert(tk.END, "\n")
        
        # 5個の型への適合率
        self.result_text.insert(tk.END, "【5個の型への適合率】\n")
        sorted_types = sorted(game_type_percentages.items(), key=lambda x: x[1], reverse=True)
        for game_type, percentage in sorted_types:
            self.result_text.insert(tk.END, f"  {game_type}: {percentage:.2f}%\n")
        self.result_text.insert(tk.END, "\n")
        
        # 共通印象スコア
        self.result_text.insert(tk.END, "【共通印象スコア】\n")
        sorted_impressions = sorted(common_impression_scores.items(), key=lambda x: x[1], reverse=True)
        for game_type, score in sorted_impressions:
            self.result_text.insert(tk.END, f"  {game_type}: {score:.3f}\n")
        self.result_text.insert(tk.END, "\n")
        
        # レコメンドされるゲーム
        self.result_text.insert(tk.END, "【レコメンドされるゲーム】\n")
        if recommended_games:
            for i, (game_title, score) in enumerate(recommended_games, 1):
                self.result_text.insert(tk.END, f"  {i}. {game_title} (スコア: {score:.3f})\n")
        else:
            self.result_text.insert(tk.END, "  レコメンドされるゲームが見つかりませんでした。\n")
        
        self.result_text.insert(tk.END, "="*80 + "\n\n")
    
    def save_result_to_csv(self, emotional_similarities: Dict, game_type_percentages: Dict,
                          common_impression_scores: Dict, recommended_games: List):
        """結果をCSVに保存"""
        try:
            results = []
            
            # 主観感情の適合度
            for emotion, similarity in emotional_similarities.items():
                results.append({
                    'カテゴリ': '主観感情',
                    '項目': emotion,
                    '値': similarity
                })
            
            # 型の適合率
            for game_type, percentage in game_type_percentages.items():
                results.append({
                    'カテゴリ': '型',
                    '項目': game_type,
                    '値': percentage
                })
            
            # 共通印象スコア
            for game_type, score in common_impression_scores.items():
                results.append({
                    'カテゴリ': '共通印象',
                    '項目': game_type,
                    '値': score
                })
            
            # レコメンドゲーム
            for i, (game_title, score) in enumerate(recommended_games, 1):
                results.append({
                    'カテゴリ': 'レコメンドゲーム',
                    '項目': f'{i}. {game_title}',
                    '値': score
                })
            
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.output_path.get(), index=False, encoding='utf-8-sig')
            
            self.result_text.insert(tk.END, f"結果を '{self.output_path.get()}' に保存しました。\n")
        
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 結果の保存中にエラーが発生しました: {str(e)}\n")


def main():
    """メイン関数"""
    root = tk.Tk()
    app = SmartGameRecommendationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

