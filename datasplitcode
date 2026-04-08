import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/taseabolt/CS620Project/main/2.0%20CLEANED-Lottery-Mega-Millions-Winning-Numbers-Beginning-2002.csv")
print("Original loaded data file:", '\n', df.head(), '\n')

# Convert dates and sort chronologically
df["Draw Date"] = pd.to_datetime(df["Draw Date"], format="%m/%d/%y")
df = df.sort_values("Draw Date").reset_index(drop=True)
print("Data file w/converted dates & ascending order:", '\n', df.head(), '\n')

# Split winning numbers into separate columns
balls = df["Winning Numbers"].str.split(expand=True).astype(int)
balls.columns = ["Num1", "Num2", "Num3", "Num4", "Num5"]
df = df.drop(columns=["Winning Numbers"])
df = pd.concat([df, balls], axis=1)
print("Data file w/winning numbers in separate columns:", '\n', df.head(), '\n')

# Split 15% val, 15% test, 70% train
train_val, test = train_test_split(df, test_size=0.15, shuffle=False)
train, val = train_test_split(train_val, test_size=0.15, shuffle=False)
print(f"Val:   {len(val)} rows")
print(f"Test:  {len(test)} rows")
print(f"Train: {len(train)} rows")

# Save split data files
val.to_csv("lotteryval.csv",     index=False)
test.to_csv("lotterytest.csv",   index=False)
train.to_csv("lotterytrain.csv", index=False)

print("Data split completed.")
