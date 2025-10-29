import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List

# --- 1. 特徴量抽出関数 (アノテーション用CSV生成に特化) ---
def create_annotation_ready_features(input_csv_path: str, time_window_sec: float = 30.0) -> pd.DataFrame:
    """
    生のキーログCSVを読み込み、時間窓ごとに特徴量を計算し、
    アノテーションとセッションIDの入力待ち状態のDataFrameを返します。
    """
    
    print(f"--- 1. 生データ '{input_csv_path}' を読み込み、特徴量を計算中 ---")
    
    try:
        log_data = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{input_csv_path}' が見つかりません。")
        return pd.DataFrame()
    
    # 列名の統一 (ご提示のデータ形式に合わせています)
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
        print(f"❌ データの型変換エラー: {e}")
        return pd.DataFrame()

    max_time = log_data['elapsed_time'].max()
    segments_list: List[Dict[str, Any]] = []
    
    start_time = 0.0
    while start_time < max_time + time_window_sec:
        end_time = start_time + time_window_sec
        window_data = log_data[(log_data['elapsed_time'] >= start_time) & (log_data['elapsed_time'] < end_time)]
        
        # 5つの特徴量を計算
        if len(window_data) == 0:
            feature_vector = {
                'Duration_Mean': 0.0,
                'Delay_Mean': time_window_sec,
                'APM': 0.0,
                'Duration_Var': 0.0,
                'Stop_Ratio': 1.0
            }
        else:
            duration_mean = window_data['duration'].mean()
            delay_mean = window_data['delay'].mean()
            apm = len(window_data) / time_window_sec * 60
            duration_var = window_data['duration'].var()
            if np.isnan(duration_var): duration_var = 0.0
            
            total_active_duration = window_data['duration'].sum()
            stop_time_ratio = (time_window_sec - total_active_duration) / time_window_sec
            stop_time_ratio = max(0.0, min(1.0, stop_time_ratio))

            feature_vector = {
                'Duration_Mean': duration_mean,
                'Delay_Mean': delay_mean,
                'APM': apm,
                'Duration_Var': duration_var,
                'Stop_Ratio': stop_time_ratio
            }
        
        # 補助情報とプレースホルダーを追加
        segment_data = {
            'Time_Start_s': round(start_time, 2),
            'Time_End_s': round(end_time, 2),
            **feature_vector,
            'Session_ID': 'S001_PLACEHOLDER',
            'True_Emotion': 'ENTER_EMOTION_HERE'
        }
        segments_list.append(segment_data)
        
        start_time = end_time

    df_features = pd.DataFrame(segments_list)
    return df_features

# --- 2. メイン実行ブロック ---
if __name__ == '__main__':
    
    # 入力ファイルパス: ご提示のキーログファイル
    INPUT_CSV_PATH = 'keydata/keylog_output.csv'
    # 出力ファイルパス: アノテーションを入力するためのCSV
    OUTPUT_CSV_PATH = 'HMMleran/features_ready_for_annotation.csv'
    
    # 実行
    df_output = create_annotation_ready_features(INPUT_CSV_PATH)
    
    if not df_output.empty:
        # 補助的なカラムを先頭に移動し、手動入力しやすい形式にする
        cols = ['Session_ID', 'True_Emotion', 'Time_Start_s', 'Time_End_s'] + \
               [col for col in df_output.columns if col not in ['Session_ID', 'True_Emotion', 'Time_Start_s', 'Time_End_s']]
        df_output = df_output[cols]
        
        df_output.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\n✅ データ '{OUTPUT_CSV_PATH}' の生成が完了しました。")
        print("\n--- 次のステップ ---")
        print("生成されたCSVファイルを開き、'Session_ID' と 'True_Emotion' のカラムにデータを手動で入力してください。")