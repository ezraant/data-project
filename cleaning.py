import pandas as pd

#Load CSV into a DataFrame
df = pd.read_csv('PlayerStatistics (1).csv', low_memory=False)

#convert 'gameDateTimeEst' to datetime objects
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])

#Filter out data from before september 2003
df_after_2004 = df[df['gameDateTimeEst'] >= '2003-09-01']

#only include the columns we need
columns_to_keep = ['personId', 'firstName', 'lastName', 'gameDateTimeEst', 'playerteamName', 'opponentteamName', 'gameType', 'home', 
                   'numMinutes', 'points', 'steals', 'reboundsTotal', 'assists', 'blocks', 'fieldGoalsAttempted', 'threePointersMade']
df_filtered = df_after_2004[columns_to_keep]

#removing preseason games
df_filtered = df_filtered[df_filtered['gameType'] != 'Preseason']
df_filtered = df_filtered[df_filtered['gameType'] != 'All-Star Game']

#removing rows with 0 minutes played
df_filtered['numMinutes'] = pd.to_numeric(df_filtered['numMinutes'], errors='coerce')
df_filtered = df_filtered[df_filtered['numMinutes'] > 0]


#Make a set of personId for anyone who has played since September 2023 so we can remove anyone not in the set
players_since_sept_2023_df = df_filtered[df_filtered['gameDateTimeEst'] >= '2023-09-01']
person_ids_since_sept_2023 = set(players_since_sept_2023_df['personId'].unique())

#make a condition for rows before September 2023
before_sept_2023_condition = df_filtered['gameDateTimeEst'] < '2023-09-01'

#make a condition for personId not in the set
not_in_set_condition = ~df_filtered['personId'].isin(person_ids_since_sept_2023)

# Combine the conditions to identify rows to be removed
rows_to_remove_condition = before_sept_2023_condition & not_in_set_condition

# Filter the DataFrame, keeping only the rows that do NOT meet the removal condition
df_final_filtered = df_filtered[~rows_to_remove_condition].copy()


#list of columns to convert to numbers
stats_cols = ['points', 'steals', 'reboundsTotal', 'assists', 'blocks', 'threePointersMade']
for col in stats_cols:  
    df_final_filtered[col] = pd.to_numeric(df_final_filtered[col], errors='coerce').fillna(0)

#making statPerMinute columns
df_final_filtered['pointsPerMinute'] = (df_final_filtered['points'] / df_final_filtered['numMinutes']).round(5)
df_final_filtered['stealsPerMinute'] = (df_final_filtered['steals'] / df_final_filtered['numMinutes']).round(5)
df_final_filtered['reboundsPerMinute'] = (df_final_filtered['reboundsTotal'] / df_final_filtered['numMinutes']).round(5)
df_final_filtered['assistsPerMinute'] = (df_final_filtered['assists'] / df_final_filtered['numMinutes']).round(5)
df_final_filtered['blocksPerMinute'] = (df_final_filtered['blocks'] / df_final_filtered['numMinutes']).round(5)
df_final_filtered['threesPerMinute'] = (df_final_filtered['threePointersMade'] / df_final_filtered['numMinutes']).round(5)

df_final_filtered.to_csv('UpdatedPlayerStatistics (1).csv', index=False)
