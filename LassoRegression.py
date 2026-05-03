import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso # Import Lasso
from sklearn.metrics import mean_absolute_error


# ============================================================
#  SETTINGS (feel free to tweak these)
# ============================================================

FILE_PATH      = "UpdatedPlayerStatistics.csv" # Changed to load the CSV with per-minute stats
LOOKBACK_GAMES = 20   # how many recent games to average over
LASSO_ALPHA    = 0.1  # Alpha parameter for Lasso regularization


# ============================================================
#  Step 1: Load the data ONCE at the start
#  (We only do this once so it's fast when you look up players)
# ============================================================

print("=" * 55)
print("   🏀  NBA OVER/UNDER PREDICTOR (Lasso Version with Enhanced Features)")
print("=" * 55)
print("\nLoading data... (this takes a few seconds)")

df_lasso = pd.read_csv(FILE_PATH)
df_lasso["gameDateTimeEst"] = pd.to_datetime(df_lasso["gameDateTimeEst"])
df_lasso["fullName"] = df_lasso["firstName"] + " " + df_lasso["lastName"]

print(f"Ready! Loaded {len(df_lasso):,} game rows.\n")


# ============================================================
#  Step 2: Define a function that runs the model
#
#  A function is like a recipe — you define it once,
#  then you can call it over and over with different inputs.
# ============================================================

def predict_player_lasso(player_name, stat, line):
    """
    Given a player name, a stat, and an over/under line,
    trains a Lasso model on their history and prints a prediction.
    """

    # --- Filter to just this player ---
    player_df = df_lasso[df_lasso["fullName"] == player_name].copy()

    if len(player_df) == 0:
        print(f"\n  Could not find '{player_name}' in the data.")
        print("  Double-check the spelling (e.g. 'LeBron James', 'Stephen Curry')\n")
        return   # stop the function here and go back to the menu

    # Sort oldest to newest for rolling averages
    player_df = player_df.sort_values("gameDateTimeEst", ascending=True).reset_index(drop=True)

    print(f"\n  Found {len(player_df)} games for {player_name}  "
          f"({player_df['gameDateTimeEst'].min().date()} to "
          f"{player_df['gameDateTimeEst'].max().date()})")

    # --- Build rolling average features ---
    # Enhanced feature_cols to include per-minute stats
    feature_cols = ["points", "reboundsTotal", "assists",
                    "numMinutes", "fieldGoalsAttempted",
                    "pointsPerMinute", "stealsPerMinute", "reboundsPerMinute",
                    "assistsPerMinute", "blocksPerMinute", "threesPerMinute"]

    for col in feature_cols:
        player_df[f"avg_{col}"] = (
            player_df[col]
            .shift(1)
            .rolling(window=LOOKBACK_GAMES, min_periods=3)
            .mean()
        )

    player_df["is_home"] = player_df["home"]

    # One-hot encode opponentteamName to factor in opponent name
    # Create the dummies as a separate dataframe
    dummies = pd.get_dummies(player_df['opponentteamName'], prefix='opp', drop_first=True)

    # Join the dummies back to the original dataframe (keeping the original column)
    player_df = pd.concat([player_df, dummies], axis=1)

    player_df = player_df.dropna().reset_index(drop=True)

    if len(player_df) < 10:
        print(f"  Not enough game history to make a prediction (need at least 10 games).\n")
        return

    # --- Split into train / test (80% / 20%) ---
    # Dynamically build input_features to include new rolling avg features and one-hot encoded opponent names
    input_features = [f"avg_{col}" for col in feature_cols] + ["is_home"]
    opponent_cols = [col for col in player_df.columns if col.startswith('opp_')]
    input_features.extend(opponent_cols)

    X = player_df[input_features]
    y = player_df[stat]

    split_index = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- Train the model (Lasso) ---
    model = Lasso(alpha=LASSO_ALPHA)
    model.fit(X_train, y_train)

    # --- Evaluate accuracy ---
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # --- Predict next game ---
    latest_input   = X.iloc[[-1]]
    predicted_stat = model.predict(latest_input)[0]

    # --- Print results ---
    print()
    print("  " + "=" * 48)
    print(f"  PREDICTION  ->  {player_name.upper()}")
    print("  " + "=" * 48)
    print(f"  Stat          : {stat}")
    print(f"  Predicted     : {predicted_stat:.1f}")
    print(f"  O/U Line      : {line}")
    print(f"  Model error   : +/-{mae:.1f} {stat}")
    print()

    gap = abs(predicted_stat - line)
    if gap < mae:
        print(f"  TOO CLOSE TO CALL  (gap of {gap:.1f} is within model error)")
    elif predicted_stat > line:
        print(f"  TAKE THE OVER   (+{predicted_stat - line:.1f} above the line)")
    else:
        print(f"  TAKE THE UNDER  ({line - predicted_stat:.1f} below the line)")

    # --- Show last 5 predictions vs actual ---
    print()
    print("  Last 5 test games (predicted vs actual):")
    comparison = player_df.iloc[split_index:].copy()
    comparison["predicted"] = y_pred_test.round(1)
    comparison["actual"]    = y_test.values
    comparison["error"]     = (comparison["predicted"] - comparison["actual"]).round(1)
    cols = ["gameDateTimeEst", "opponentteamName", "actual", "predicted", "error"]
    print(comparison[cols].tail(5).to_string(index=False))
    print()


# ============================================================
#  Step 3: The main loop
#  "while True" means: keep asking forever until they type quit
# ============================================================

VALID_STATS = {
    "1": "points",
    "2": "reboundsTotal",
    "3": "assists",
    "4": "steals",
    "5": "blocks",
    "6": "threePointersMade"
}

print("Type 'quit' at any prompt to exit.\n")

while True:

    # --- Ask for player name ---
    print("-" * 55)
    player_input = input("  Player name (e.g. LeBron James): ").strip()

    if player_input.lower() == "quit":
        print("\nGoodbye!\n")
        break

    # --- Ask for stat ---
    print()
    print("  Which stat?")
    print("    1 -> Points")
    print("    2 -> Rebounds")
    print("    3 -> Assists")
    print("    4 -> Steals")
    print("    5 -> Blocks")
    print("    6 -> Three Pointers Made")
    stat_input = input("  Enter 1, 2, 3, 4, 5, or 6: ").strip()

    if stat_input.lower() == "quit":
        print("\nGoodbye!\n")
        break

    if stat_input not in VALID_STATS:
        print("  Please enter 1, 2, 3, 4, 5, or 6.\n")
        continue

    stat = VALID_STATS[stat_input]

    # --- Ask for the over/under line ---
    line_input = input(f"  Over/Under line for {stat} (e.g. 24.5): ").strip()

    if line_input.lower() == "quit":
        print("\nGoodbye!\n")
        break

    # Convert the line to a number — catch typos
    try:
        line = float(line_input)
    except ValueError:
        print("  That doesn't look like a number. Try something like 24.5\n")
        continue

    # --- Run the prediction! ---
    predict_player_lasso(player_input, stat, line) # Call the Lasso version of the function
