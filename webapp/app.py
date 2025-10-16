import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# 既存処理をインポート
from kenkyu.Getcsv import export_reviews_to_csv
from kenkyu.Readcsv import score_reviews_csv

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret')

# 保存先ディレクトリ
BASE_DIR = "/Users/sakumasin/code/code_try/kenkyu"
REVIEWS_DIR = os.path.join(BASE_DIR, "webapp", "data", "reviews")
SCORES_DIR = os.path.join(BASE_DIR, "webapp", "data", "scores")
os.makedirs(REVIEWS_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)

# Word2Vecモデルの既定パス（環境変数で上書き可）
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', "/Users/sakumasin/Documents/vscode/zemi/models/word2vec/entity_vector/entity_vector.model.bin")

@app.get('/')
def index():
    return redirect(url_for('step1'))

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    if request.method == 'POST':
        app_id = request.form.get('app_id', '').strip()
        output_name = request.form.get('output_name', '').strip()
        review_type = request.form.get('review_type', 'positive')
        language = request.form.get('language', 'japanese')
        num_reviews = int(request.form.get('num_reviews', '100'))
        if not app_id or not output_name:
            flash('アプリIDと出力CSV名は必須です')
            return redirect(url_for('step1'))
        filename = secure_filename(output_name)
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        output_path = os.path.join(REVIEWS_DIR, filename)
        try:
            export_reviews_to_csv(app_id, output_path, review_type=review_type, num_reviews=num_reviews, language=language, api_key=os.environ.get('STEAM_API_KEY'))
            flash(f'レビューCSVを作成しました: {output_path}')
            return redirect(url_for('step2', reviews_csv=filename))
        except Exception as e:
            flash(f'エラー: {e}')
            return redirect(url_for('step1'))
    return render_template('step1.html')

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    # 前段で作ったCSV名をクエリから取得可能
    preset_reviews_csv = request.args.get('reviews_csv')
    if request.method == 'POST':
        reviews_csv = request.form.get('reviews_csv', '').strip()
        output_name = request.form.get('output_name', '').strip()
        model_path = request.form.get('model_path', DEFAULT_MODEL_PATH).strip()
        if not reviews_csv or not output_name:
            flash('評定するCSVと出力CSV名は必須です')
            return redirect(url_for('step2'))
        input_path = os.path.join(REVIEWS_DIR, reviews_csv)
        if not os.path.isfile(input_path):
            flash('指定のレビューCSVが見つかりません')
            return redirect(url_for('step2'))
        filename = secure_filename(output_name)
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        output_path = os.path.join(SCORES_DIR, filename)
        try:
            score_reviews_csv(input_path, output_path, model_path=model_path)
            flash(f'評定スコアCSVを作成しました: {output_path}')
            return render_template('done.html', scores_csv=filename, abs_path=output_path)
        except Exception as e:
            flash(f'エラー: {e}')
            return redirect(url_for('step2'))
    # レビューCSVの一覧を表示用に取得
    existing_reviews = [f for f in os.listdir(REVIEWS_DIR) if f.lower().endswith('.csv')]
    return render_template('step2.html', preset_reviews_csv=preset_reviews_csv, existing_reviews=existing_reviews, default_model_path=DEFAULT_MODEL_PATH)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)


