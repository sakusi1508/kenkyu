import requests
import pandas as pd

def get_steam_reviews(app_id, review_type='all', num_reviews=10, language='all', api_key='86B191F42E308C0E8245F62118A53A04'):
    base_url = f'http://store.steampowered.com/appreviews/{app_id}?json=1'
    params = {
        'filter': review_type,
        'num_per_page': num_reviews,
        'language': language,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        reviews = response.json()
        return reviews
    else:
        print(f'レビューの取得に失敗しました: {response.status_code}, {response.text}')

# 使用例
app_id = 2357570 # 使用するSteamのアプリID
api_key = '86B191F42E308C0E8245F62118A53A04'  # 自分のSteam APIキー
reviews = get_steam_reviews(app_id, review_type='positive', num_reviews=10, language='japanese', api_key=api_key)

# 修正した部分: DataFrameの作成
df = pd.DataFrame(reviews['reviews'])  # レスポンスの'reviews'キーにレビューのデータが含まれていると仮定

# CSVファイルに保存
df.to_csv('/Users/sakumasin/Documents/vscode/zemi/csv/OV2_2_review.csv', index=False)
print('csv完了')
