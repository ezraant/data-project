# ============================================================
#  NBA Player Over/Under Predictor
#  Predicts: Points, Rebounds, Assists, Steals, Blocks, Threes
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# ============================================================
#  SETTINGS
# ============================================================

FILE_PATH      = "UpdatedPlayerStatistics (1).csv"
RECENT_SEASONS = 4
RIDGE_ALPHA    = 50
WEIGHT_POWER   = 3.0


# ============================================================
#  Step 1: Load data once
# ============================================================

print("=" * 55)
print("   NBA OVER/UNDER PREDICTOR")
print("   Predicts Points, Rebounds, Assists,")
print("   Steals, Blocks & Threes!")
print("=" * 55)
print("\nLoading data... (this takes a few seconds)")

df = pd.read_csv(FILE_PATH)
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
df["fullName"] = df["firstName"] + " " + df["lastName"]
valid_players = df["fullName"].unique().tolist()

all_teams = sorted(df["opponentteamName"].dropna().unique().tolist())

print(f"Ready! Loaded {len(df):,} game rows.\n")


# ============================================================
#  All 6 stats the user can predict
# ============================================================

STATS = {
    "1": ("points",          "Points"),
    "2": ("reboundsTotal",   "Rebounds"),
    "3": ("assists",         "Assists"),
    "4": ("steals",          "Steals"),
    "5": ("blocks",          "Blocks"),
    "6": ("threePointersMade", "Threes Made"),
}


# ============================================================
#  Step 2: Build features
# ============================================================

def build_features(player_df, stat):

    counting_cols = [
        "points", "reboundsTotal", "assists",
        "numMinutes", "fieldGoalsAttempted",
        "steals", "blocks", "threePointersMade"
    ]

    per_minute_cols = [
        "pointsPerMinute", "stealsPerMinute", "reboundsPerMinute",
        "assistsPerMinute", "blocksPerMinute", "threesPerMinute"
    ]

    all_feature_cols = counting_cols + per_minute_cols

    # Rolling averages at 3 window sizes
    for w in [3, 5, 10]:
        for col in all_feature_cols:
            player_df[f"avg{w}_{col}"] = (
                player_df[col].shift(1).rolling(w, min_periods=2).mean()
            )

    # Trend: hot or cold streak?
    for col in all_feature_cols:
        player_df[f"trend_{col}"] = (
            player_df[f"avg3_{col}"] - player_df[f"avg10_{col}"]
        )

    # Lag features: actual raw values from last 3 games
    for lag in [1, 2, 3]:
        player_df[f"lag{lag}_{stat}"] = player_df[stat].shift(lag)

    # Per-minute lag for the predicted stat
    stat_per_min = stat + "PerMinute"
    if stat_per_min in player_df.columns:
        for lag in [1, 2, 3]:
            player_df[f"lag{lag}_{stat_per_min}"] = player_df[stat_per_min].shift(lag)

    player_df["is_home"] = player_df["home"]
    player_df = player_df.dropna().reset_index(drop=True)

    return player_df, all_feature_cols


# ============================================================
#  Step 3: Run Ridge regression for one stat
# ============================================================

def predict_stat(player_clean, stat, is_home, all_feature_cols):

    stat_per_min = stat + "PerMinute"
    per_min_lags = (
        [f"lag{lag}_{stat_per_min}" for lag in [1, 2, 3]]
        if stat_per_min in player_clean.columns else []
    )

    input_features = (
        [f"avg{w}_{col}" for w in [3, 5, 10] for col in all_feature_cols] +
        [f"trend_{col}"  for col in all_feature_cols] +
        [f"lag{lag}_{stat}" for lag in [1, 2, 3]] +
        per_min_lags +
        ["is_home"]
    )
    input_features = [c for c in input_features if c in player_clean.columns]

    X = player_clean[input_features]
    y = player_clean[stat]

    split   = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    weights = np.linspace(1.0, WEIGHT_POWER, len(X_train))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=RIDGE_ALPHA))
    ])
    model.fit(X_train, y_train, reg__sample_weight=weights)

    y_pred_test  = model.predict(X_test)
    mae          = mean_absolute_error(y_test, y_pred_test)

    next_game            = X.iloc[[-1]].copy()
    next_game["is_home"] = int(is_home)
    predicted            = model.predict(next_game)[0]

    return predicted, mae


# ============================================================
#  Step 4: Full prediction for selected stats
# ============================================================

def predict_player(player_name, selected_stats, lines, opponent, is_home):

    player_df = df[df["fullName"] == player_name].copy()

    if len(player_df) == 0:
        print(f"\n  Could not find '{player_name}' in the data.")
        print("  Double-check spelling (e.g. 'LeBron James', 'Stephen Curry')\n")
        return

    player_df = player_df.sort_values("gameDateTimeEst", ascending=True).reset_index(drop=True)
    cutoff    = player_df["gameDateTimeEst"].max() - pd.DateOffset(years=RECENT_SEASONS)
    player_df = player_df[player_df["gameDateTimeEst"] >= cutoff].reset_index(drop=True)

    print(f"\n  Found {len(player_df)} games for {player_name} "
          f"(last {RECENT_SEASONS} seasons: "
          f"{player_df['gameDateTimeEst'].min().date()} to "
          f"{player_df['gameDateTimeEst'].max().date()})")

    all_player   = df[df["fullName"] == player_name]
    games_vs_opp = all_player[all_player["opponentteamName"] == opponent]

    print()
    print("  " + "=" * 48)
    print(f"  PREDICTIONS  ->  {player_name.upper()}")
    print(f"  Opponent: {opponent}  ({'Home' if is_home else 'Away'})")
    print("  " + "=" * 48)

    for stat, label in selected_stats:
        line = lines[stat]

        player_copy          = player_df.copy()
        player_clean, feats  = build_features(player_copy, stat)

        if len(player_clean) < 20:
            print(f"\n  {label}: Not enough data\n")
            continue

        predicted, mae = predict_stat(player_clean, stat, is_home, feats)

        # Context stats
        last3_avg  = player_clean[stat].iloc[-3:].mean()
        avg_vs_opp = games_vs_opp[stat].mean() if len(games_vs_opp) > 0 else None
        home_away  = player_clean[player_clean["home"] == int(is_home)][stat].mean()

        gap = abs(predicted - line)
        if gap < mae:
            call = "⚠️  TOO CLOSE TO CALL"
        elif predicted > line:
            call = f"✅  TAKE THE OVER   (+{predicted - line:.1f})"
        else:
            call = f"❌  TAKE THE UNDER  ({line - predicted:.1f} below)"

        print()
        print(f"  {'─' * 46}")
        print(f"  {label.upper()}")
        print(f"  {'─' * 46}")
        print(f"  O/U Line        : {line}")
        print(f"  Predicted       : {predicted:.1f}")
        print(f"  Model error     : +/-{mae:.1f}")
        print(f"  Last 3 game avg : {last3_avg:.1f}")
        if avg_vs_opp is not None:
            print(f"  Avg vs {opponent:<12}: {avg_vs_opp:.1f}  ({len(games_vs_opp)} games)")
        print(f"  {'Home' if is_home else 'Away'} avg (recent): {home_away:.1f}")
        print(f"  Recommendation  : {call}")

    print()

# stat-lookup
# if you want to use this not for betting but to look up stats from the database..
# or if you want to do your own betting research and look up stats yourself..
def lookup_stats(player_name):
    player_df = df[df["fullName"] == player_name].copy()
    if player_df.empty:
        print(f"\n  Could not find '{player_name}'.")
        return

    # --- ADD THIS LOGIC HERE ---
    # Create a 'Season' column: If month >= 10, season is year + 1. Otherwise, it's just the year.
    player_df['Season'] = player_df['gameDateTimeEst'].apply(
        lambda x: x.year + 1 if x.month >= 10 else x.year
    )

    print(f"\n--- STAT LOOKUP: {player_name.upper()} ---")
    print("1: Season Averages (e.g., 2026)")
    print("2: Averages vs Specific Opponent")
    choice = input("Choice: ").strip()

    stats_cols = ["numMinutes", "points", "reboundsTotal", "assists", "steals", "blocks", "threePointersMade"]

    if choice == "1":
        year_in = input("Enter Season Year (e.g., 2026 for 25-26 season): ").strip().lower()
        
        if year_in == 'current':
            # Use the most recent game's calculated season
            target_season = player_df['Season'].max()
        else:
            try:
                target_season = int(year_in)
            except:
                print("Invalid year format."); return
        
        # Filter by the new 'Season' column instead of the calendar year
        filtered = player_df[player_df["Season"] == target_season]
        label = f"{target_season-1}-{str(target_season)[2:]} Season"
    
    elif choice == "2":
        opp = input("Enter Opponent Name: ").strip()
        scope = input("1: This Season | 2: Career vs Opponent: ").strip()
        
        if scope == "1":
            this_season = player_df['Season'].max()
            filtered = player_df[(player_df["opponentteamName"] == opp) & 
                                 (player_df["Season"] == this_season)]
            label = f"vs {opp} ({this_season-1}-{str(this_season)[2:]} Season)"
        else:
            filtered = player_df[player_df["opponentteamName"] == opp]
            label = f"vs {opp} (Career)"
    else: return

    if filtered.empty:
        print(f"\n  No games found for the {label}.")
    else:
        print(f"\n  {label} ({len(filtered)} games)")
        avgs = filtered[stats_cols].mean()
        for s, v in avgs.items():
            display_name = "Minutes Per Game" if s == "numMinutes" else s
            print(f"  {s:<18}: {v:.1f}")

# ============================================================
#  Step 5: Main loop
# ============================================================

print("Type 'quit' at any prompt to exit.\n")

while True:

    print("=" * 55)
    print("  MAIN MENU")
    print("  1-> Betting Predictions (Ridge Model)")
    print("  2-> Stat Lookup (Averages/History)")
    print("=" * 55)

    mode = input("Select Mode: ").strip().lower()
    if mode in ["quit", "q", "exit"]: break

    if mode == "1":
        # --- Player ---
        player_input = input("  Player name (e.g. LeBron James): ").strip()
        if player_input.lower() == "quit":
            print("\nGoodbye!\n")
            break
        if player_input not in valid_players:
            print(f"\n   ⚠️ Error: '{player_input}' not found in database.")
            print("   Check spelling and capitalization (e.g., 'Jalen Brunson').")
            continue

        # --- Opponent ---
        print(f"\n  Available teams: {', '.join(all_teams)}")
        opponent_input = input("  Opponent team name (e.g. Celtics): ").strip()
        if opponent_input.lower() == "quit":
            print("\nGoodbye!\n")
            break
        if opponent_input not in all_teams:
            print(f"\n  Could not find '{opponent_input}'. Check the team list above.\n")
            continue

        # --- Home or away ---
        location_input = input("  Home or Away? (h/a): ").strip().lower()
        if location_input == "quit":
            print("\nGoodbye!\n")
            break
        if location_input not in ["h", "a"]:
            print("  Please type h for home or a for away.\n")
            continue
        is_home = location_input == "h"

        # --- Pick which stats to predict ---
        print()
        print("  Which stats do you want predictions for?")
        for key, (stat, label) in STATS.items():
            print(f"    {key} -> {label}")
        print("  Enter numbers separated by commas (e.g. 1,2,3 or 1,4,5,6)")
        stat_input = input("  Your choice: ").strip()

        if stat_input.lower() == "quit":
            print("\nGoodbye!\n")
            break

        # Parse which stats the user picked
        chosen_keys = [s.strip() for s in stat_input.split(",")]
        selected_stats = []
        valid = True
        for key in chosen_keys:
            if key not in STATS:
                print(f"  '{key}' is not a valid option. Please pick from 1-6.\n")
                valid = False
                break
            selected_stats.append(STATS[key])  # (stat_col, label)

        if not valid:
            continue

        # --- Get the O/U line for each selected stat ---
        print()
        lines = {}
        for stat, label in selected_stats:
            line_input = input(f"  O/U line for {label} (e.g. 2.5): ").strip()
            if line_input.lower() == "quit":
                print("\nGoodbye!\n")
                valid = False
                break
            try:
                lines[stat] = float(line_input)
            except ValueError:
                print(f"  That doesn't look like a number. Try again.\n")
                valid = False
                break

        if not valid:
            break

        # --- Run predictions ---
        predict_player(player_input, selected_stats, lines, opponent_input, is_home)
    elif mode == "2":
        name = input("\nPlayer Name: ").strip()
        if name.lower() in ["quit", "q"]: break
        lookup_stats(name)