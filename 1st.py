import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file into a Pandas dataframe
df = pd.read_csv('data.csv')

# Show basic statistical information about the data
print(df.describe())

# Check if there are any missing values in the data
print("Missing values:", df.isnull().sum().sum())

# Replace missing values with the mean
df = df.fillna(df.mean())

# Calculate minimum, maximum, count, and mean of two columns, "Calories" and "Duration"
agg_data = df[["Calories", "Duration"]].agg(["min", "max", "count", "mean"])
print(agg_data)

# Filter rows based on conditions
df_cal_500_1000 = df[(df["Calories"] >= 500) & (df["Calories"] <= 1000)]
print("the rows with calories values between 500 and 1000\n",df_cal_500_1000)
df_cal_gt_500_pulse_lt_100 = df[(df["Calories"] > 500) & (df["Pulse"] < 100)]
print("the rows with calories values > 500 and pulse < 100\n",df_cal_gt_500_pulse_lt_100)

# Create a new dataframe with a column dropped
df_modified = df.drop("Maxpulse", axis=1)

# Drop the column from the original dataframe
df.drop("Maxpulse", axis=1, inplace=True)

# Change the datatype of a column
df["Calories"] = df["Calories"].astype(int)

# Plot the data
plt.scatter(df["Duration"], df["Calories"])
plt.xlabel("Duration")
plt.ylabel("Calories")
plt.show()
