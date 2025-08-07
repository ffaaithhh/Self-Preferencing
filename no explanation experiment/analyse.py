import pandas as pd

# Load the CSV file
df = pd.read_csv('no explanation experiment/google_responses.csv')

# Count how many times Google's product is chosen
google_wins = (df['Response'] == df['Google']).sum()

print(f"Google's product was chosen {google_wins} times out of {len(df)} comparisons.")