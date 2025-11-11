#!/bin/bash
# ゲームレコメンデーションシステム GUIアプリケーション起動スクリプト

# プロジェクトルートディレクトリに移動
cd "$(dirname "$0")/.."

# Pythonスクリプトを実行
python DB/game_recommendation_gui.py

