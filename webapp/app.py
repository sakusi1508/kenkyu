import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# プロジェクト設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 依存関係のインポート
try:
    from Getcsv import export_reviews_to_csv
    from Readcsv import score_reviews_csv
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要な依存関係をインストールしてください: pip install pandas requests MeCab gensim scikit-learn numpy")
    sys.exit(1)

# Flaskアプリケーション初期化
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限

# 環境設定
os.environ.setdefault('MECABRC', '/opt/homebrew/etc/mecabrc')

# ディレクトリ設定
BASE_DIR = PROJECT_ROOT
REVIEWS_DIR = os.path.join(BASE_DIR, "webapp", "data", "reviews")
SCORES_DIR = os.path.join(BASE_DIR, "webapp", "data", "scores")
os.makedirs(REVIEWS_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)

# デフォルト設定
DEFAULT_MODEL_PATH = os.environ.get(
    'MODEL_PATH', 
    "/Users/sakumasin/Documents/vscode/zemi/models/word2vec/entity_vector/entity_vector.model.bin"
)

# ユーティリティ関数
def _validate_form_data(data, required_fields):
    """フォームデータの検証"""
    missing = [field for field in required_fields if not data.get(field, '').strip()]
    return missing

def _create_csv_filename(name):
    """CSVファイル名の作成"""
    filename = secure_filename(name)
    return filename if filename.lower().endswith('.csv') else f"{filename}.csv"

@app.get('/')
def index():
    return redirect(url_for('step1'))

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    if request.method == 'POST':
        form_data = {
            'app_id': request.form.get('app_id', '').strip(),
            'output_name': request.form.get('output_name', '').strip(),
            'review_type': request.form.get('review_type', 'positive'),
            'language': request.form.get('language', 'japanese'),
            'num_reviews': int(request.form.get('num_reviews', '100'))
        }
        
        # 必須フィールドの検証
        missing = _validate_form_data(form_data, ['app_id', 'output_name'])
        if missing:
            flash('アプリIDと出力CSV名は必須です')
            return redirect(url_for('step1'))
        
        # ファイル名とパスの作成
        filename = _create_csv_filename(form_data['output_name'])
        output_path = os.path.join(REVIEWS_DIR, filename)
        
        try:
            export_reviews_to_csv(
                form_data['app_id'], 
                output_path,
                review_type=form_data['review_type'],
                num_reviews=form_data['num_reviews'],
                language=form_data['language'],
                api_key=os.environ.get('STEAM_API_KEY')
            )
            flash(f'レビューCSVを作成しました: {filename}')
            return redirect(url_for('step2', reviews_csv=filename))
        except Exception as e:
            flash(f'エラー: {e}')
            return redirect(url_for('step1'))
    
    return render_template('step1.html')

def _get_existing_reviews():
    """既存のレビューCSVファイル一覧を取得"""
    try:
        return [f for f in os.listdir(REVIEWS_DIR) if f.lower().endswith('.csv')]
    except Exception:
        return []

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    preset_reviews_csv = request.args.get('reviews_csv')
    
    if request.method == 'POST':
        form_data = {
            'reviews_csv': request.form.get('reviews_csv', '').strip(),
            'output_name': request.form.get('output_name', '').strip(),
            'model_path': request.form.get('model_path', DEFAULT_MODEL_PATH).strip()
        }
        
        # 必須フィールドの検証
        missing = _validate_form_data(form_data, ['reviews_csv', 'output_name'])
        if missing:
            flash('評定するCSVと出力CSV名は必須です')
            return redirect(url_for('step2'))
        
        # 入力ファイルの存在確認
        input_path = os.path.join(REVIEWS_DIR, form_data['reviews_csv'])
        if not os.path.isfile(input_path):
            flash('指定のレビューCSVが見つかりません')
            return redirect(url_for('step2'))
        
        # 出力ファイル名とパスの作成
        filename = _create_csv_filename(form_data['output_name'])
        output_path = os.path.join(SCORES_DIR, filename)
        
        try:
            score_reviews_csv(input_path, output_path, model_path=form_data['model_path'])
            return render_template('done.html', scores_csv=filename, abs_path=output_path)
        except Exception as e:
            flash(f'エラー: {e}')
            return redirect(url_for('step2'))
    
    # GET リクエストの場合
    existing_reviews = _get_existing_reviews()
    return render_template(
        'step2.html', 
        preset_reviews_csv=preset_reviews_csv, 
        existing_reviews=existing_reviews, 
        default_model_path=DEFAULT_MODEL_PATH
    )

if __name__ == '__main__':
    # 本番環境ではdebug=Falseに設定
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)


