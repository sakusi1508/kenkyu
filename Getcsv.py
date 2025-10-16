import os
import requests
import pandas as pd

def get_steam_reviews(app_id, review_type='all', num_reviews=10, language='all', api_key=None):
    """Steamレビューを取得する"""
    base_url = f'http://store.steampowered.com/appreviews/{app_id}?json=1'
    params = {
        'filter': review_type,
        'num_per_page': num_reviews,
        'language': language
    }
    if api_key:
        params['key'] = api_key
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    raise RuntimeError(f'レビューの取得に失敗しました: {response.status_code}, {response.text}')

def export_reviews_to_csv(app_id, output_csv_path, review_type='all', num_reviews=10, language='all', api_key=None):
    """Steamレビューを取得してCSVに保存する
    
    Args:
        app_id: SteamアプリID
        output_csv_path: 出力CSVの絶対パス
        review_type: フィルタ（'all'|'positive'|'negative'）
        num_reviews: 取得件数
        language: 言語（'all'|'japanese'）
        api_key: Steam APIキー（オプション）
    
    Returns:
        保存されたCSVの絶対パス
    """
    reviews = get_steam_reviews(app_id, review_type, num_reviews, language, api_key)
    
    if 'reviews' not in reviews:
        raise ValueError("APIレスポンスに 'reviews' キーが見つかりませんでした")
    
    df = pd.DataFrame(reviews['reviews'])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path

if __name__ == '__main__':
    # 簡易CLI実行: 環境変数やデフォルト値から取得
    app_id = os.environ.get('APP_ID', '2357570')
    api_key = os.environ.get('STEAM_API_KEY')
    output_csv_path = os.environ.get('OUTPUT_CSV', '/tmp/reviews.csv')
    export_reviews_to_csv(app_id, output_csv_path, review_type='positive', num_reviews=10, language='japanese', api_key=api_key)
    print('csv完了:', output_csv_path)
