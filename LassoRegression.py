import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

# ============================================================
#  SETTINGS
# ============================================================
FILE_PATH      = "UpdatedPlayerStatistics (1).csv"
WINDOW_LONG    = 20    
WINDOW_SHORT   = 5     
LASSO_ALPHA    = 0.01  # Lowered for more sensitivity

# ============================================================
#  Step 1: Load Data
# ============================================================
print("=" * 55)
print("   🏀  NBA PREDICTOR: FULL STAT MOMENTUM EDITION")
print("=" * 55)
print("\nLoading data...")

df_lasso = pd.read_csv(FILE_PATH)
df_lasso["gameDateTimeEst"] = pd.to_datetime(df_lasso["gameDateTimeEst"])
df_lasso["fullName"] = df_lasso["firstName"] + " " + df_lasso["lastName"]

# Global One-Hot Encoding for opponents
df_lasso = pd.get_dummies(df_lasso, columns=['opponentteamName'], prefix='opp', drop_first=False)

print(f"Ready! Data loaded.\n")

# ============================================================
#  Step 2: The Prediction Engine
# ============================================================
def predict_player_lasso(player_name, stat, line, next_opp, is_home_val):
    # Filter to player
    player_df = df_lasso[df_lasso["fullName"] == player_name].copy()

    if len(player_df) == 0:
        print(f"\n  Could not find '{player_name}'. Check spelling!")
        return

    player_df = player_df.sort_values("gameDateTimeEst").reset_index(drop=True)

    # --- Feature Engineering ---
    base_cols = ["points", "reboundsTotal", "assists", "steals", "blocks", "threePointersMade",
                 "numMinutes", "pointsPerMinute", "stealsPerMinute", "reboundsPerMinute", 
                 "assistsPerMinute", "blocksPerMinute", "threesPerMinute"]

    feature_cols = []
    for col in base_cols:
        # Long-term average (Baseline)
        player_df[f"avg20_{col}"] = player_df[col].shift(1).rolling(window=WINDOW_LONG, min_periods=5).mean()
        # Short-term momentum (Hot Streak)
        player_df[f"hot5_{col}"] = player_df[col].shift(1).rolling(window=WINDOW_SHORT, min_periods=2).mean()
        feature_cols.extend([f"avg20_{col}", f"hot5_{col}"])

    player_df["is_home"] = player_df["home"]
    player_df = player_df.dropna().reset_index(drop=True)

    if len(player_df) < 15:
        print(f"  Not enough history for {player_name} (need 15+ games).")
        return

    # Build Input Features
    opponent_cols = [col for col in df_lasso.columns if col.startswith('opp_')]
    input_features = feature_cols + ["is_home"] + opponent_cols

    X = player_df[input_features].fillna(0)
    y = player_df[stat]

    # Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train Model
    model = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
    model.fit(X_train, y_train)

    # Prediction for Next Game
    latest_input = pd.DataFrame(0, index=[0], columns=input_features)
    
    # Use the absolute latest game's rolling stats to predict the upcoming one
    for col in feature_cols:
        latest_input[col] = player_df[col].iloc[-1]
    
    latest_input["is_home"] = is_home_val
    opp_col = f"opp_{next_opp}"
    
    if opp_col not in latest_input.columns:
        print(f"  Warning: '{next_opp}' not recognized. Using league average defense.")
    else:
        latest_input[opp_col] = 1

    predicted_stat = model.predict(latest_input)[0]
    mae = mean_absolute_error(y_test, model.predict(X_test))

    # Output Results
    print("\n  " + "=" * 48)
    print(f"  PREDICTION -> {player_name.upper()} vs {next_opp.upper()}")
    print(f"  Location: {'HOME' if is_home_val == 1 else 'AWAY'}")
    print("  " + "=" * 48)
    print(f"  Stat      : {stat}")
    print(f"  Predicted : {predicted_stat:.1f}")
    print(f"  O/U Line  : {line}")
    print(f"  Model MAE : +/-{mae:.1f}")

    gap = abs(predicted_stat - line)
    if gap < (mae * 0.4): # Higher confidence threshold
        print(f"\n  BETTING ADVICE: TOO CLOSE TO CALL")
    elif predicted_stat > line:
        print(f"\n  BETTING ADVICE: TAKE THE OVER (+{predicted_stat - line:.1f})")
    else:
        print(f"\n  BETTING ADVICE: TAKE THE UNDER ({line - predicted_stat:.1f})")

# ============================================================
#  Step 3: User Loop
# ============================================================
STATS_MAP = {
    "1": "points", 
    "2": "reboundsTotal", 
    "3": "assists",
    "4": "steals",
    "5": "blocks",
    "6": "threePointersMade"
}

while True:
    print("-" * 55)
    name = input("Player Name: ").strip()
    if name.lower() == 'quit': break
    
    print("\n1:Pts | 2:Reb | 3:Ast | 4:Stl | 5:Blk | 6:3PM")
    stat_choice = input("Select Stat: ").strip()
    if stat_choice not in STATS_MAP: continue
    
    opp = input("Opponent (e.g., Lakers): ").strip()
    loc = input("Is player at HOME? (y/n): ").strip().lower()
    is_home = 1 if loc == 'y' else 0
    
    line_in = input("O/U Line: ").strip()
    try:
        line = float(line_in)
        predict_player_lasso(name, STATS_MAP[stat_choice], line, opp, is_home)
    except ValueError:
        print("Invalid number.")