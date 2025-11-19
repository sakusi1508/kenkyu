
import os
import sys
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_recommender import recommend_games, load_game_scores

def test_14_factor_recommendation():
    # Path to merged_game_scores.csv
    game_scores_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'syuyougametitle_score', 'merged_game_scores.csv')
    print(f"Testing with path: {game_scores_path}")
    
    # 1. Test loading
    try:
        df = load_game_scores(game_scores_path)
        print("\nLoaded DataFrame:")
        print(df[['game_title']].head().to_string())
        print(f"Total rows: {len(df)}")
        
        # Check for 14 factors
        expected_factors = ['RPG性', 'アクション性', 'パズル性', '報酬の程度'] # Check a few
        missing = [f for f in expected_factors if f not in df.columns]
        if missing:
            print(f"Error: Missing factors in CSV: {missing}")
            return
            
    except Exception as e:
        print(f"Error during loading: {e}")
        return

    # 2. Test recommendation
    # Mock user percentages (e.g., prefers Fantasy and RPG)
    user_percentages = {
        'C1:ファンタジー型': 40.0,
        'C2:映画型': 10.0,
        'C3:現実活動型': 10.0,
        'C4:キャラクター型': 10.0,
        'C5:シミュレーション型': 30.0
    }
    
    # Mock common impression scores
    common_impression_scores = {
        'C1:ファンタジー型': 0.8,
        'C2:映画型': 0.2,
        'C3:現実活動型': 0.2,
        'C4:キャラクター型': 0.2,
        'C5:シミュレーション型': 0.6
    }
    
    print("\nRunning recommendation...")
    try:
        recommendations = recommend_games(
            game_type_percentages=user_percentages,
            common_impression_scores=common_impression_scores,
            game_scores_path=game_scores_path,
            top_n=5
        )
        
        print("\nTop 5 Recommendations:")
        for i, (game, score) in enumerate(recommendations, 1):
            print(f"{i}. {game} (Score: {score:.4f})")
            
    except Exception as e:
        print(f"Error during recommendation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_14_factor_recommendation()
