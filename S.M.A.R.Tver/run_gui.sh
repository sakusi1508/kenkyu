#!/bin/bash
# S.M.A.R.T. ゲームレコメンデーションシステムの起動スクリプト

# プロジェクトルートディレクトリに移動
cd "$(dirname "$0")/.."

# Pythonスクリプトを実行
python S.M.A.R.Tver/smart_gui.py

