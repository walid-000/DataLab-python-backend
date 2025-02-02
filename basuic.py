import pandas as pd

# Create a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [88, 92, 85, 90]
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Calculate the average score
average_score = df['Score'].mean()
print("\nAverage Score:", average_score)

# Filter rows where age is greater than 30
filtered_df = df[df['Age'] > 30]
print("\nFiltered DataFrame (Age > 30):")
print(filtered_df)

# Add a new column: 'Passed' based on the score
df['Passed'] = df['Score'] >= 90
print("\nDataFrame with 'Passed' column:")
print(df)

# Group by 'Passed' and calculate the average age
grouped_data = df.groupby('Passed')['Age'].mean()
print("\nAverage Age by Pass/Fail:")
print(grouped_data)
