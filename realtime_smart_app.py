
import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import numpy as np
import joblib
from pynput import keyboard

# --- Path Configuration ---
# Allow importing from sibling directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Add project root to path to find DB, etc.
PROJECT_ROOT = CURRENT_DIR
# If this script is run from root, PROJECT_ROOT is fine. 
# If run from inside a folder, we might need to go up.
# Assuming this file is placed at the root of the repo as requested.

# Add S.M.A.R.Tver to path to allow importing modules inside it
SMART_DIR = os.path.join(CURRENT_DIR, 'S.M.A.R.Tver')
if SMART_DIR not in sys.path:
    sys.path.append(SMART_DIR)

# Import project modules
try:
    # Import from S.M.A.R.Tver (added to path above)
    from keylog_analyzer import extract_features_from_keylog, estimate_factors_from_keylog_features
    from emotion_converter import convert_to_emotional_labels, convert_to_game_types, compare_emotional_and_type_similarity, FACTOR_LIST
    
    # Import from DB (in PROJECT_ROOT)
    from DB.recommendation_system import recommend_games, HMM_EMOTIONAL_STATES, load_emotion_game_mapping
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Define fallback or dummy imports if strictly necessary for testing,
    # but for production it should fail.
    # We will define HMM_EMOTIONAL_STATES to avoid NameError later if DB import failed but others didn't.
    HMM_EMOTIONAL_STATES = ['Sensory', 'Brain', 'Healing', 'Setting'] 
    print("Please make sure you are running this script from the project root or correct paths are set.") 

# --- Constants & Configuration ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DEFAULT_ANALYSIS_WINDOW = 30  # seconds
HMM_MODEL_PATH = os.path.join(CURRENT_DIR, 'HMMleran', 'hmm_emotion_model_4states.pkl')
UPDATE_INTERVAL_MS = 5000 # 5 seconds

# --- Keylogger Logic (Background Thread) ---
class KeyMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_keys = {} # key -> press_time
        self.log_data = [] # List of dicts: {key, press_time, release_time, duration, delay}
        self.last_release_time = None
        self.listener = None
        self.running = False
        self.start_time = time.time()

    def start(self):
        if self.running:
            return
        self.running = True
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        print("KeyMonitor started.")

    def stop(self):
        if self.running and self.listener:
            self.listener.stop()
            self.listener = None
            self.running = False
            print("KeyMonitor stopped.")

    def _get_key_str(self, key):
        try:
            return key.char
        except AttributeError:
            return str(key).replace('Key.', '')

    def on_press(self, key):
        key_str = self._get_key_str(key)
        now = time.time()
        
        with self.lock:
            # Ignore auto-repeat
            if key_str in self.active_keys:
                return
            
            self.active_keys[key_str] = now
            
            # Delay calculation (time since last release)
            delay = 0.0
            if self.last_release_time is not None:
                delay = max(0.0, now - self.last_release_time)

            # Note: We don't log incomplete presses yet, we log on release to get duration
            # But for realtime visualization, we might want to know it's pressed.
            # For extraction features, we need duration, so we wait for release.

    def on_release(self, key):
        key_str = self._get_key_str(key)
        now = time.time()
        
        with self.lock:
            if key_str in self.active_keys:
                press_time = self.active_keys.pop(key_str)
                duration = now - press_time
                
                # Retrieve last release to calc delay (if this wasn't the very first key)
                # Note: 'delay' is normally time from PREVIOUS release to CURRENT press
                # We calculated it at press time, but didn't store it. Let's recalculate or store locally?
                # Actually, simpler to just track last_release_time globally.
                
                delay = 0.0
                if self.last_release_time is not None:
                     delay = max(0.0, press_time - self.last_release_time)
                
                self.last_release_time = now
                
                # Elapsed time from script start
                elapsed = now - self.start_time

                entry = {
                    'key_info': key_str,
                    'press_time': press_time,
                    'release_time': now,
                    'duration': duration,
                    'delay': delay,
                    'elapsed_time': elapsed
                }
                self.log_data.append(entry)

                # Prune old data to keep memory usage low? 
                # For now, maybe keep all or prune very old (> 1 hour?)
                # We only need last X seconds for analysis.
                # Let's prune periodically in get_snapshot or separately.

    def get_recent_data(self, window_seconds=30):
        """
        Returns a DataFrame of key logs from (now - window_seconds) to now.
        """
        now = time.time()
        cutoff = now - window_seconds
        
        with self.lock:
            # Filter relevant data
            # Optimization: since log_data is appended in time order, we could binary search or slice.
            # For simplicity, list comprehension is fine for typical key typing speeds.
            recent_logs = [
                entry for entry in self.log_data 
                if entry['release_time'] >= cutoff
            ]
            
            # Optional: Prune very old data (older than window * 2) to prevent infinite growth
            if len(self.log_data) > 10000:
                prune_cutoff = now - (window_seconds * 10)
                self.log_data = [e for e in self.log_data if e['release_time'] >= prune_cutoff]

        if not recent_logs:
            return pd.DataFrame(columns=['key_info', 'press_time', 'release_time', 'duration', 'delay', 'elapsed_time'])

        return pd.DataFrame(recent_logs)
    
    def get_last_n_logs(self, n=10):
        """Returns string representation of last n logs for display."""
        with self.lock:
            slice_data = self.log_data[-n:]
        
        lines = []
        for d in slice_data:
            lines.append(f"Key: {d['key_info']:<10} | Dur: {d['duration']:.3f}s | Delay: {d['delay']:.3f}s")
        return "\n".join(reversed(lines)) # Newest top

# --- Main GUI Application ---
class RealtimeSmartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Emotion & Game Recommendation System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # State
        self.analysis_window = tk.IntVar(value=DEFAULT_ANALYSIS_WINDOW)
        self.monitor = KeyMonitor()
        self.monitor.start()
        
        # HMM Model
        self.hmm_model = self.load_hmm_model()
        self.hmm_state_names = HMM_EMOTIONAL_STATES
        # Try to load state names file if exists
        state_file = HMM_MODEL_PATH.replace('.pkl', '_states.pkl')
        if os.path.exists(state_file):
            try:
                self.hmm_state_names = joblib.load(state_file)
            except:
                pass
        
        self.game_mapping = None
        try:
            self.game_mapping = load_emotion_game_mapping()
        except:
            print("Warning: Game mapping CSV not found.")

        # Layout
        self._setup_ui()
        
        # Start Loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(1000, self.update_loop)

    def load_hmm_model(self):
        if os.path.exists(HMM_MODEL_PATH):
            try:
                model = joblib.load(HMM_MODEL_PATH)
                print(f"Loaded HMM model from {HMM_MODEL_PATH}")
                return model
            except Exception as e:
                print(f"Failed to load HMM model: {e}")
        else:
            print(f"HMM model not found at {HMM_MODEL_PATH}")
        return None

    def _setup_ui(self):
        # Main Container (PanedWindow for Left/Right split)
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Left Panel: Analysis & Emotion ---
        left_frame = tk.Frame(paned, width=400)
        paned.add(left_frame)
        
        # Control Header
        ctrl_frame = tk.LabelFrame(left_frame, text="設定 / Settings")
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(ctrl_frame, text="Time Window (sec):").pack(side=tk.LEFT, padx=5)
        tk.Spinbox(ctrl_frame, from_=5, to=300, textvariable=self.analysis_window, width=5).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(ctrl_frame, text="Monitoring...", fg="green")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Current Emotion Box
        emo_frame = tk.LabelFrame(left_frame, text="推定感情 / Estimated Emotion", font=("Arial", 14, "bold"))
        emo_frame.pack(fill=tk.X, pady=10, ipady=10)
        
        self.current_emotion_var = tk.StringVar(value="Waiting for data...")
        tk.Label(emo_frame, textvariable=self.current_emotion_var, font=("Arial", 24, "bold"), fg="blue").pack(expand=True)
        
        self.confidence_var = tk.StringVar(value="")
        tk.Label(emo_frame, textvariable=self.confidence_var, font=("Arial", 10)).pack()

        # Factor Visualization
        factor_frame = tk.LabelFrame(left_frame, text="14因子 / 14 Factors")
        factor_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        canvas_container = tk.Frame(factor_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.factor_progress_bars = {}
        row = 0
        col = 0
        max_rows = 7
        
        # Use a grid for factors
        for factor in FACTOR_LIST:
            f_label = tk.Label(canvas_container, text=factor, anchor="w", width=15)
            f_label.grid(row=row, column=col*2, padx=5, pady=2, sticky="w")
            
            pb = ttk.Progressbar(canvas_container, length=100, mode='determinate', maximum=1.0)
            pb.grid(row=row, column=col*2+1, padx=5, pady=2)
            self.factor_progress_bars[factor] = pb
            
            row += 1
            if row >= max_rows:
                row = 0
                col += 1
        
        # --- Right Panel: Recommendations ---
        right_frame = tk.Frame(paned)
        paned.add(right_frame)
        
        rec_label = tk.Label(right_frame, text="おすすめゲーム / Recommended Games", font=("Arial", 14, "bold"))
        rec_label.pack(pady=5)
        
        # Listbox with Scrollbar
        list_frame = tk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.rec_listbox = tk.Listbox(list_frame, font=("Arial", 12), yscrollcommand=scrollbar.set)
        self.rec_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.rec_listbox.yview)

        # --- Bottom Panel: Key Monitor (Debug) ---
        bottom_frame = tk.Frame(self.root, height=150)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        bottom_frame.pack_propagate(False) # Fixed height
        
        tk.Label(bottom_frame, text="Realtime Key Monitor (Recent Inputs)", font=("Arial", 10, "bold"), anchor="w").pack(fill=tk.X)
        
        self.log_display = scrolledtext.ScrolledText(bottom_frame, state='disabled', font=("Courier", 10))
        self.log_display.pack(fill=tk.BOTH, expand=True)

    def on_close(self):
        self.monitor.stop()
        self.root.destroy()
        print("Application closed.")

    def update_loop(self):
        # 1. Update Monitor Text
        recent_log_text = self.monitor.get_last_n_logs(15)
        self.log_display.config(state='normal')
        self.log_display.delete(1.0, tk.END)
        self.log_display.insert(tk.END, recent_log_text)
        self.log_display.config(state='disabled')
        
        # 2. Analyze Data
        window = self.analysis_window.get()
        df = self.monitor.get_recent_data(window)
        
        if not df.empty and len(df) > 5: # Require at least some keystrokes
            self.status_label.config(text="Analyzing...", fg="blue")
            try:
                self.perform_analysis(df, window)
            except Exception as e:
                print(f"Analysis error: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.config(text="Error in Analysis", fg="red")
        else:
            self.status_label.config(text="Waiting for more input...", fg="orange")

        # Schedule next update
        self.root.after(UPDATE_INTERVAL_MS, self.update_loop)

    def perform_analysis(self, df, window_sec):
        # Normalize timestamps so the window starts at 0 for the analyzer
        # The analyzer expects data to start from 0 and chunks it by window_sec.
        if not df.empty:
            start_time = df['elapsed_time'].min()
            df = df.copy()
            df['elapsed_time'] = df['elapsed_time'] - start_time
        
        X, lengths = extract_features_from_keylog(df, time_window_sec=window_sec)
        
        # If extraction produced multiple segments (e.g. slight overflow), take the last valid one 
        # or the one that represents the bulk of the data. 
        # Usually with normalized 0-30s data and 30s window, we get 1 or 2 segments.
        # We want the aggregate features of this window.
        if X.shape[0] > 1:
            # If multiple segments, average them or just take the first (main) one?
            # extract_features loops: 0-30, 30-60.
            # If data is 0.0 to 29.9, we get 1 segment.
            # If data is 0.0 to 30.1, we might get 2.
            # We care about the features of the current 'window'. 
            # Ideally we pick the one with more data or average.
            # For simplicity, let's take the mean of the features if it split.
            feature_vector = np.mean(X, axis=0).reshape(1, -1)
            X = feature_vector

        estimated_emotion = None
        
        # Method 1: HMM
        if self.hmm_model and X.shape[0] > 0:
            try:
                logprob, state_sequence = self.hmm_model.decode(X, algorithm="viterbi")
                # Get last state as current emotion? Or most frequent in the sequence?
                # Usually sequence corresponds to sub-segments of the window.
                # If window is 30s and function breaks it into chunks, we get a sequence.
                # extract_features_from_keylog breaks it into ONE chunk if we pass the whole window?
                # Let's check extract_features:
                #   while start_time < max_time + time_window_sec:
                #       ...
                #       segments.append(...)
                # So it splits by time_window_sec.
                # If we pass 30s as time_window_sec, and our DF is 30s long, we get roughly 1 segment.
                
                if len(state_sequence) > 0:
                    current_state_idx = state_sequence[-1] # The latest state
                    if 0 <= current_state_idx < len(self.hmm_state_names):
                        estimated_emotion = self.hmm_state_names[current_state_idx]
            except Exception as e:
                print(f"HMM Decode Error: {e}")
        
        # Method 2: Rule-based (Fallback or hybrid)
        # also needed for factors
        factors = estimate_factors_from_keylog_features(X)
        
        # If HMM failed, use rule conversion
        if not estimated_emotion:
            emotional_similarities = convert_to_emotional_labels(factors)
            estimated_emotion = max(emotional_similarities, key=emotional_similarities.get)
            self.confidence_var.set("(Rule-based Fallback)")
        else:
            self.confidence_var.set("(HMM Estimated)")

        # C. Update UI
        self.current_emotion_var.set(estimated_emotion)
        
        # Update Factors
        for factor, val in factors.items():
            if factor in self.factor_progress_bars:
                self.factor_progress_bars[factor]['value'] = val
        
        # D. Recommendation
        # Use existing logic from DBT/recommendation_system.py
        # recommend_games_from_emotions takes (emotion_sequence, ...)
        
        # Create a mock sequence of just the current emotion (since we are "Realtime")
        # Or if we keep history, we could pass history.
        # For simplicity, just use current.
        
        try:
            # We use db function to get games
            # It expects sequence, so list of 1.
            dominant_emo, rec_games, _ = recommend_games([estimated_emotion], top_n=10)
            
            self.rec_listbox.delete(0, tk.END)
            if rec_games:
                for game in rec_games:
                    self.rec_listbox.insert(tk.END, game)
            else:
                self.rec_listbox.insert(tk.END, "No recommendations found.")
                
        except Exception as e:
            print(f"Recommendation Error: {e}")

if __name__ == "__main__":
    if not os.path.exists('HMMleran'): 
        # Just a check to warn the user if cwd is wrong
        print("Warning: HMMleran directory not found in current directory.")
        print(f"Current Working Directory: {os.getcwd()}")
    
    root = tk.Tk()
    app = RealtimeSmartApp(root)
    root.mainloop()
