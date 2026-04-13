import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('/content/sample_data/PlayerStatistics.csv')

# Convert 'gameDateTimeEst' to datetime objects
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])

# Filter out data from before the year 2004
df_filtered = df[df['gameDateTimeEst'].dt.year >= 2004]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('PlayerStatistics_after_2004.csv', index=False)

print("Filtered data saved to 'PlayerStatistics_after_2004.csv'")
