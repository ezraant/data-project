import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

FILE_PATH      = "UpdatedPlayerStatistics (1).csv"
RECENT_SEASONS = 4 #only using the past 4 seasons worth of data since very old data is unreliable to use today
RIDGE_ALPHA    = 50 #this controls how drastic the ridge regression model prevents outlier predictions
WEIGHT_POWER   = 3.0 #this is how much more we will weight recent games compared to a game a season ago (3x more)


#the next few lines are the welcome prompt for the user to breifly explain the program
print("==================================================")
print("   NBA Alternate line PREDICTOR")
print("   Predicts Points, Rebounds, Assists,")
print("   Steals, Blocks & Threes!")
print("==================================================")
print("\nLoading data, please give us a few seconds")

df = pd.read_csv(FILE_PATH)
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"]) #turns the date column into actual date format
df["fullName"] = df["firstName"] + " " + df["lastName"] # combines the first and last name columns into one column named fullName
valid_players = df["fullName"].unique().tolist()
all_teams = sorted(df["opponentteamName"].dropna().unique().tolist())

print(f"Data Loaded {len(df):,} game rows.\n") #tells the user the data is loaded and the number of games that were loaded


# ===============================================================================================================================================

STATS = { #the six stats the program will be able to make alternate line predictions on
    "1": ("points",          "Points"),
    "2": ("reboundsTotal",   "Rebounds"),
    "3": ("assists",         "Assists"),
    "4": ("steals",          "Steals"),
    "5": ("blocks",          "Blocks"),
    "6": ("threePointersMade", "Threes Made"),
}


# ===============================================================================================================================================

def build_function (player_df, stat): #this function is used multiple times when the user gives the player and the stat(s) they are interested in

    counting_columns = [ #these are the stats we will pull from the cleaned csv file to allow the program to calculate averages and alternate lines
        "points", 
        "reboundsTotal", 
        "assists",
        "numMinutes", 
        "fieldGoalsAttempted",
        "steals", 
        "blocks", 
        "threePointersMade"
    ]

    per_minute_stats_columns = [ #per minute stats for each player which helps create more reliable predicitons based on teammate injuries and other factors
        "pointsPerMinute", 
        "stealsPerMinute", 
        "reboundsPerMinute",
        "assistsPerMinute", 
        "blocksPerMinute", 
        "threesPerMinute"
    ]

    total_function_columns = counting_columns + per_minute_stats_columns #combines the two lists above into one

    # creating a rollowing average using the past 3, 5, and 10 games the player played. Then the rolling average is the average of the last x number of games. we went with 3, 5, and 10 to have a very recent window, short window, and medium window
    for rolling_average in [3, 5, 10]: #loops through all three windows of games we chose
        for col in total_function_columns: #loops through each statistic column
            player_df[f"avg{rolling_average}_{col}"] = ( #creates a new column
                player_df[col].shift(1).rolling(rolling_average, min_periods = 2).mean() #takes the column and shifts it by 1 to disregard current games and then look at the window of rolling average games and finds the average of those games
            )

    #calculates if the player trend. Trend = recent average minus broader average. If the trend is positive the player is on a hot streak and if negative the player is on a cold streak.   
    for col in total_function_columns: #iterates through every statistic
        player_df[f"trend_{col}"] = ( #creates a new column named trend
            player_df[f"avg3_{col}"] - player_df[f"avg10_{col}"] #last 3 game averages - last 10 game averages
        )

    for raw_numbers in [1, 2, 3]: #raw numbers are just the stat the player put up. 1 is the last game, 2 is 2 games ago, and 3 is 3 games ago. helps the model guage player momentum along with the hot and cold streak implementation above
        player_df[f"raw_numbers{raw_numbers}_{stat}"] = player_df[stat].shift(raw_numbers) #gets the info from the raw numbers games

    # Calculates the stats a player records per minute on average
    stat_per_min = stat + "PerMinute" #makes the column name

    if stat_per_min in player_df.columns:
        for raw_numbers in [1, 2, 3]:
            player_df[f"raw_numbers{raw_numbers}_{stat_per_min}"] = player_df[stat_per_min].shift(raw_numbers) #retreives the per-minute average from the 3 most recent games

    player_df["is_home"] = player_df["home"] #creates a column to see if it is a home or away game. 0 = away and 1 = home
    player_df = player_df.dropna().reset_index(drop=True) #removes any rows that have missing values since it shows the player is not a valid player to make predicitions on

    return player_df, total_function_columns


# ===============================================================================================================================================

def predict_player_stats(player_clean, stat, is_home, total_function_columns):

    stat_per_min = stat + "PerMinute" #makes the per-minute stat column
    per_min_lags = (
        [f"raw_numbers{raw_numbers}_{stat_per_min}" for raw_numbers in [1, 2, 3]]
        if stat_per_min in player_clean.columns else []
    )

    input_features = ( #makes the list of input features
        [f"avg{rolling_average}_{col}" for rolling_average in [3, 5, 10] for col in total_function_columns] + #rolling average stats
        [f"trend_{col}"  for col in total_function_columns] + #trends
        [f"raw_numbers{raw_numbers}_{stat}" for raw_numbers in [1, 2, 3]] +  #raw player stats
        per_min_lags +
        ["is_home"] #home or away game
    )
    input_features = [c for c in input_features if c in player_clean.columns] #only keeps columns that are present in the data to ensure everything is functioning and was built correctly

    X = player_clean[input_features] #all of the input data
    y = player_clean[stat] #output (prediction)

    #splits the data into testing and training (20% testing and 80% training)
    split = int(len(X) * 0.8) # where the split happens
    X_train = X.iloc[:split] #training input of older games
    X_test = X.iloc[split:] #testing inputs from more recent games
    y_train = y.iloc[:split] # training outputs from older games
    y_test = y.iloc[split:] #testing outputs from more recent games

    weights = np.linspace(1.0, WEIGHT_POWER, len(X_train)) #creates a weight system so more recent games have more impact in the prediction

    model = Pipeline([
        ("scaler", StandardScaler()), # scales all of the stats/features into the same range so all stat formats are the same and do not dominate others
        ("reg",    Ridge(alpha=RIDGE_ALPHA)) #regression model that adapts to the patterns as it is being used
    ])

    model.fit(X_train, y_train, reg__sample_weight=weights) #trains the model using the weights from above. Learns the relationship between the inputs and outputs

    y_testing = model.predict(X_test) #now that the model is trained, use it to predict stats/alternate lines for upcoming games
    mean_error = mean_absolute_error(y_test, y_testing) #calculate the average error to calculate how inaccurate the model is

    next_game = X.iloc[[-1]].copy() #gets the last row of the data to get the players current game and copies it so we dont change the original data
    next_game ["is_home"] = int(is_home)#converts a true/false to 1/0

    predicted = model.predict(next_game)[0] #gets the first result from the list and returns it from the model

    return predicted, mean_error


# ===============================================================================================================================================


def predict_player(player_name, selected_stats, lines, opponent, is_home):

    player_df = df[df["fullName"] == player_name].copy() #filters the enture dataset into just games of the player name inputted.

    if len(player_df) == 0: #if there are no games found user will be prompted to check spelling 
        print(f"\n  Could not find '{player_name}'.")
        print("  Please double-check spelling (e.g. 'LeBron James', 'Jalen Brunson')\n")
        return

    player_df = player_df.sort_values("gameDateTimeEst", ascending=True).reset_index(drop=True) #sorts the player's games from oldest to most recent 
    cutoff = player_df["gameDateTimeEst"].max() - pd.DateOffset(years=RECENT_SEASONS) #calculates the cut off date for the calculations
    player_df = player_df[player_df["gameDateTimeEst"] >= cutoff].reset_index(drop=True) #filters out and ignores games that are older than the calculated cutoff

    print(f"\n  Found {len(player_df)} games for {player_name} " #Tells the user the amount of valid games found in the dataset
          f"(last {RECENT_SEASONS} seasons: "
          f"{player_df['gameDateTimeEst'].min().date()} to "
          f"{player_df['gameDateTimeEst'].max().date()})")

    all_time_player_stats = df[df["fullName"] == player_name] #finds the player's all time career games
    games_vs_selected_opponent = all_time_player_stats[all_time_player_stats["opponentteamName"] == opponent] #filters to the games the player played againts the chosen opponent by the user

    #prints the results
    print()
    print("  " + "==================================================")
    print(f"  PREDICTIONS  ->  {player_name.upper()}")
    print(f"  Opponent: {opponent}  ({'Home' if is_home else 'Away'})")
    print("  " + "==================================================")

    for stat, label in selected_stats: #loops through each statistic the user requested and runs the prediction model for each selected statistic
        line = lines[stat]

        player_copy = player_df.copy() #make a copy of the player data
        player_clean, feats= build_function(player_copy, stat) #builds all of the needed features for the  specefic statistic

        if len(player_clean) < 20: #If there are less than 20 games for the statistic, skip the stat since there is not enough data to work from
            print(f"\n  {label}: Not enough data\n")
            continue

        predicted, mean_error = predict_player_stats(player_clean, stat, is_home, feats) #runs the ridge regression model and outputs the prediction and model error (if any)
        prev_3_averages  = player_clean[stat].iloc[-3:].mean() # average of the previous 3 games played
        has_played_opponent = len(games_vs_selected_opponent) > 0 #if player has played againts the chosen opponent
        average_vs_opponent = games_vs_selected_opponent[stat].mean() if has_played_opponent else None #career average againts the opponenet
        home_away_averages = player_clean[player_clean["home"] == int(is_home)][stat].mean() #averages at home compared to away
        difference = abs(predicted - line) #difference in predicted ALT line and actual line

        if difference < mean_error:
            call = " TOO CLOSE TO CALL"
        elif predicted > line:
            call = f" TAKE THE OVER  (+{predicted - line:.1f})"
        else:
            call = f" TAKE THE UNDER ({line - predicted:.1f} below)"

        #Finally prints the stats and the predictions for the chosen player, opponent, and stat(s)
        print()
        print(f"  {'=================================================='}")
        print(f"  {label.upper()}")
        print(f"  {'=================================================='}")
        print(f"  OVER/UNDER Line        : {line}")
        print(f"  Predicted       : {predicted:.1f}")
        print(f"  Model error     : +/-{mean_error:.1f}")
        print(f"  Last 3 game average : {prev_3_averages:.1f}")

        if average_vs_opponent is not None:
            print(f"  Avg vs {opponent:<12}: {average_vs_opponent:.1f}  ({len(games_vs_selected_opponent)} games)")

        print(f"  {'Home' if is_home else 'Away'} avg (recent): {home_away_averages:.1f}")
        print(f"  Recommendation  : {call}")

    print() #prints blank line to make it easier to read and make room for the next player input if the user needs information for multiple players


# stat-lookup
# if you want to use this not for betting but to look up stats from the database
# or if you want to do your own betting research and look up stats yourself
def lookup_stats(player_name):
    player_df = df[df["fullName"] == player_name].copy()
    if player_df.empty:
        print(f"\n  Could not find '{player_name}'.")
        return

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
            # use the most recent game's calculated season
            # * was using calendar year before
            target_season = player_df['Season'].max()
        else:
            try:
                target_season = int(year_in)
            except:
                print("Invalid year format."); return
        
        # season not calender year
        filtered = player_df[player_df["Season"] == target_season]
        label = f"{target_season-1}-{str(target_season)[2:]} Season"
    
    elif choice == "2":
        opp = input("Enter Opponent Name: ").strip()
        # after opponent name, have option for all time stats or just this seasons stats vs that opponent
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

# ===============================================================================================================================================

print("Type 'quit' at any prompt to exit.\n") #throughout this loop, if the user types 'quit', the program will end

while True: #starting the main loop

    print("==================================================")
    print("  menu")
    print("  1 ->betting predictions (ridge model)")
    print("  2 ->stat lookup (averages/history)")
    print("==================================================")

    mode = input("select mode: ").strip().lower()

    if mode == "quit":
        print("\nThank you for using our predictor! Goodbye.\n")
        break

    # stat-lookup
    if mode == "2":
        player_input = input("  Player Name: ").strip()
        if player_input.lower() == "quit":
            print("\nThank you for using our predictor! Goodbye.\n")
            break
        
        if player_input not in valid_players: # ensuring the player exists
            print(f"\n  ⚠️ Error: '{player_input}' not found in database.")
            continue
            
        lookup_stats(player_input)
        continue # return to main menu after lookup

   # betting predicts mode
    elif mode == "1":
        # Asks for the user to input the player name
        player_input = input("  Player Name (e.g. LeBron James): ").strip()

        if player_input.lower() == "quit":
            print("\nThank you for using our predictor! Goodbye.\n")
            break
        
        if player_input not in valid_players: # ensures that the name inputted is in the database
            print(f"\n  ⚠️ Error: '{player_input}' not found in database.")
            continue

        # Asks the useer for the desired opponent for the player they inputted
        print(f"\n  Available Teams: {', '.join(all_teams)}")

        opponent_input = input("  Opponent Team Name (e.g. Knicks): ").strip()

        if opponent_input.lower() == "quit":
            print("\nThank you for using our predictor! Goodbye.\n")
            break

        if opponent_input not in all_teams: #ensures that the team the user inputted is in the NBA
            print(f"\n  Could not find '{opponent_input}'. Please refer to the team list above.\n") #if the user inputs an invalid team name
            continue

        # Ask the user if it is a home or away game
        location_input = input("  Home or Away Game? (h/a): ").strip().lower()

        if location_input == "quit":
            print("\nThank you for using our predictor! Goodbye.\n")
            break

        if location_input not in ["h", "a"]: #if the user does not input an h (home) or a (away)
            print("  Please type an 'h' for a home game or an 'a' for an away game.\n")
            continue

        is_home = location_input == "h"

        # Prompt the user to input any number(s) 1-6 to indicate what stat they are interested in
        print()

        print("Which stat(s) do you want predictions for?")

        for key, (stat, label) in STATS.items(): #loop to display the menu of options for the user
            print(f"{key} -> {label}")

        print("  Enter numbers separated by commas (e.g. 1,2,3 or 1,3,5,)")

        stat_input = input(" Your choice: ").strip()

        if stat_input.lower() == "quit":
            print("\nThank you for using our predictor! Goodbye.\n")
            break

        
        selected_category = [s.strip() for s in stat_input.split(",")] #converts the input into ["1", "3" "5"] if the user did "1,3,5"
        selected_stats = [] #holds the stat column and its label in a tuple

        valid = True
        for key in selected_category: #loops through each category/number the user selected
            if key not in STATS:
                print(f"  '{key}' is not a valid option. Please pick from 1-6.\n")
                valid = False
                break

            selected_stats.append(STATS[key])  # add the (stat_col, label) tuple to the list

        if not valid:
            continue

        print()

        lines = {}
        valid = True

        for stat, label in selected_stats: #loops through every stat category the user chose
            line_input = input(f"  O/U line for {label} (e.g. 4.5): ").strip() #prompts the user to input the actual over/under line
            
            if line_input.lower() == "quit":
                print("\nThank you for using our predictor! Goodbye.\n")
                valid = False
                break

            try:
                lines[stat] = float(line_input) #converts the input into an actual number/float

            except ValueError: #occurs if the value inputted is not a valid number
                print(f"  That doesn't look like a valid number or input. Please try again.\n")
                valid = False
                break

        if not valid:
            continue # changed to continue to return to menu instead of breaking

        predict_player(player_input, selected_stats, lines, opponent_input, is_home)
    
    else:
        print("  Invalid mode selection. Please type 1 or 2.")