import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('PlayerStatistics.csv', low_memory=False)

# Convert 'gameDateTimeEst' to datetime objects
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])

# Filter out data from before september 2003
df_after_2004 = df[df['gameDateTimeEst'] >= '2003-09-01']

#only include the columns we need: personId, firstName, lastName, gameDateTimeEst, playerteamName, opponentteamName, gameType, home, numMinutes, points, steals, reboundsTotal, assists, blocks, fieldGoalsAttempted, threePointersMade
columns_to_keep = ['personId', 'firstName', 'lastName', 'gameDateTimeEst', 'playerteamName', 'opponentteamName', 'gameType', 'home', 'numMinutes', 'points', 'steals', 'reboundsTotal', 'assists', 'blocks', 'fieldGoalsAttempted', 'threePointersMade']
df_filtered = df_after_2004[columns_to_keep]

df_filtered = df_filtered[df_filtered['gameType'] != 'Preseason']
# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('PlayerStatistics_after_2004.csv', index=False)

#removing preseason games
print("Filtered data saved to 'PlayerStatistics_after_2004.csv'")

#removing rows with 0 minutes played
df_filtered['numMinutes'] = pd.to_numeric(df_filtered['numMinutes'], errors='coerce')
df_filtered = df_filtered[df_filtered['numMinutes'] > 0]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000) # no new lines 
# lebron_recent = df_filtered[(df_filtered['firstName'] == 'LeBron') & (df_filtered['lastName'] == 'James')]
# print(lebron_recent.tail())

# Assuming df_filtered is already loaded and processed as in the previous step

# 1. Make a set of personId for anyone who has played since September 2023
players_since_sept_2023_df = df_filtered[df_filtered['gameDateTimeEst'] >= '2023-09-01']
person_ids_since_sept_2023 = set(players_since_sept_2023_df['personId'].unique())

# 2. Remove any rows prior to that date where the personId value is not in that set
# Create a condition for rows before September 2023
before_sept_2023_condition = df_filtered['gameDateTimeEst'] < '2023-09-01'

# Create a condition for personId not in the set
not_in_set_condition = ~df_filtered['personId'].isin(person_ids_since_sept_2023)

# Combine the conditions to identify rows to be removed
rows_to_remove_condition = before_sept_2023_condition & not_in_set_condition

# Filter the DataFrame, keeping only the rows that do NOT meet the removal condition
df_final_filtered = df_filtered[~rows_to_remove_condition]

print(f"Original df_filtered shape: {df_filtered.shape}")
print(f"Players who played since Sept 2023: {len(person_ids_since_sept_2023)}")
print(f"Final filtered DataFrame shape: {df_final_filtered.shape}")

# Display the first few rows of the final filtered DataFrame
print("\nLast 5 rows of df_final_filtered:")
print(df_final_filtered.tail())

df_final_filtered.to_csv('ActivePlayerStatistics_after_2004.csv', index=False)