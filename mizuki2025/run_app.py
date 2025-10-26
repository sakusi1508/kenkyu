#!/usr/bin/env python3
"""
感情分析キーロガーアプリケーションの起動スクリプト
"""

import sys
import os
import subprocess

def check_dependencies():
    """必要な依存関係をチェック"""
    required_packages = [
        'pandas', 'numpy', 'pynput', 'hmmlearn', 'sklearn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 以下のパッケージが不足しています:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n以下のコマンドでインストールしてください:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """メイン関数"""
    print("=== 感情分析キーロガーアプリケーション ===")
    print()
    
    # 依存関係のチェック
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ 依存関係のチェック完了")
    print()
    
    # アプリケーションの起動
    try:
        from emotion_keylogger_app import EmotionKeyloggerApp
        import tkinter as tk
        
        print("🚀 アプリケーションを起動中...")
        root = tk.Tk()
        app = EmotionKeyloggerApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ アプリケーションの起動に失敗しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
