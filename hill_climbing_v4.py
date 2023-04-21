import pandas as pd

# Load the dataset
dataset = pd.read_csv('Cleaned_Advertisements.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)

# Split dataset into training and testing datasets
train_data = dataset[:700]
test_data = dataset[700:]

# Create a dictionary to store the correlation values of each attribute with ClickedOnAd
correlation_dict = {}

# Calculate the correlation of each attribute with ClickedOnAd
for column in dataset.columns:
    if column == 'ClickedOnAd':
        continue
    correlation = dataset['ClickedOnAd'].corr(dataset[column])
    correlation_dict[column] = correlation

# Sort the attributes based on absolute correlation with ClickedOnAd
sorted_attributes = sorted(correlation_dict.items(), key=lambda x: abs(x[1]), reverse=True)
print("Sorted attributes based on absolute correlation with ClickedOnAd: ", sorted_attributes)

# Identify the most important attribute
most_important_attr = sorted_attributes[0][0]

# Create groups of 10 based on the most important attribute
test_data['group'] = pd.cut(test_data[most_important_attr], bins=range(int(test_data[most_important_attr].min()), int(test_data[most_important_attr].max())+11, 10), labels=False)

# Calculate the number of true values of ClickedOnAd for each group and sort the groups
group_counts = test_data.groupby('group')['ClickedOnAd'].sum().sort_values(ascending=False).to_dict()

# Sort the testing dataset based on the sorted attributes and unique values
test_data = test_data.reindex(columns=[column[0] for column in sorted_attributes])
test_data = test_data.sort_values(by=[column[0] for column in sorted_attributes])

# Create groups of 10 based on the most important attribute
test_data['group'] = pd.cut(test_data[most_important_attr], bins=range(int(test_data[most_important_attr].min()), int(test_data[most_important_attr].max())+11, 10), labels=False)

# Assign a rank to each group based on the number of true values of ClickedOnAd and sort the test data
test_data['rank'] = test_data['group'].apply(lambda x: sorted(list(group_counts.keys()), key=lambda y: -group_counts[y]).index(x)+1)
test_data = test_data.sort_values(['group', 'rank'])

# Write the sorted test data to a new csv file
test_data.to_csv('sorted_test_data.csv', index=False)
