import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('PlayerStatistics.csv')

# Convert 'gameDateTimeEst' to datetime objects
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])

# Filter out data from before the year 2004
df_after_2004 = df[df['gameDateTimeEst'] >= '2003-09-01']

#only include the columns we need: personId, firstName, lastName, gameDateTimeEst, playerteamName, opponentteamName, gameType, home, numMinutes, points, steals, reboundsTotal, assists, blocks, fieldGoalsAttempted, threePointersMade
columns_to_keep = ['personId', 'firstName', 'lastName', 'gameDateTimeEst', 'playerteamName', 'opponentteamName', 'gameType', 'home', 'numMinutes', 'points', 'steals', 'reboundsTotal', 'assists', 'blocks', 'fieldGoalsAttempted', 'threePointersMade']
df_filtered = df_after_2004[columns_to_keep]


# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('PlayerStatistics_after_2004.csv', index=False)

print("Filtered data saved to 'PlayerStatistics_after_2004.csv'")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # no new lines 
lebron_recent = df_filtered[(df_filtered['firstName'] == 'LeBron') & (df_filtered['lastName'] == 'James')]
print(lebron_recent.tail())