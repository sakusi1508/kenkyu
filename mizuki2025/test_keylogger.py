#!/usr/bin/env python3
"""
キーロガーのテストスクリプト
macOSでの権限問題を診断します
"""

import sys
import time
from pynput import keyboard

def test_keylogger():
    """キーロガーの基本機能をテスト"""
    print("=== キーロガーテスト開始 ===")
    print("注意: このテストは5秒間実行されます")
    print("何かキーを押してみてください...")
    print()
    
    key_count = 0
    
    def on_press(key):
        nonlocal key_count
        key_count += 1
        try:
            print(f"キー押下: {key.char if hasattr(key, 'char') and key.char else key}")
        except AttributeError:
            print(f"キー押下: {key}")
    
    def on_release(key):
        try:
            print(f"キー解放: {key.char if hasattr(key, 'char') and key.char else key}")
        except AttributeError:
            print(f"キー解放: {key}")
        
        # ESCキーで終了
        if key == keyboard.Key.esc:
            print("ESCキーが押されました。テストを終了します。")
            return False
    
    try:
        # キーロガーリスナーを開始
        print("キーロガーリスナーを開始中...")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            print("✅ キーロガーリスナーが正常に開始されました")
            print("5秒間待機中...")
            time.sleep(5)
            print("⏰ 5秒経過しました")
            
        print(f"✅ テスト完了。記録されたキー数: {key_count}")
        return True
        
    except Exception as e:
        print(f"❌ キーロガーテストに失敗しました: {e}")
        print("\n考えられる原因:")
        print("1. macOSのアクセシビリティ権限が不足している")
        print("2. ターミナルにアクセシビリティ権限が与えられていない")
        print("3. pynputライブラリの問題")
        print("\n解決方法:")
        print("1. システム環境設定 > セキュリティとプライバシー > プライバシー > アクセシビリティ")
        print("2. ターミナル（またはPython）にアクセシビリティ権限を付与")
        print("3. アプリケーションを再起動")
        return False

if __name__ == "__main__":
    success = test_keylogger()
    sys.exit(0 if success else 1)


