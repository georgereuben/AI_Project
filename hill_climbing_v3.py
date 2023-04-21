import pandas as pd

# Load the dataset
dataset = pd.read_csv('Cleaned_Advertisements.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)

# Split dataset into training and testing datasets
train_data = dataset[:700]
test_data = dataset[700:]

print(test_data.columns)

# Create a dictionary to store the correlation values of each attribute with ClickedOnAd
correlation_dict = {}

# Calculate the correlation of each attribute with ClickedOnAd
for column in dataset.columns:
    if column == 'ClickedOnAd':
        continue
    correlation = dataset['ClickedOnAd'].corr(dataset[column])
    correlation_dict[column] = correlation

# Sort the attributes based on correlation with ClickedOnAd             #    ALSO ADD ABSOLUTE VALUE OF CORRELATION HERE
sorted_attributes = sorted(correlation_dict.items(), key=lambda x: abs(x[1]), reverse=True)
print("Sorted attributes based on absolute correlation with ClickedOnAd: ", sorted_attributes)

# Create a dictionary to store the number of true values for each unique value of each attribute
unique_values_dict = {}

# Calculate the number of true values for each unique value of each attribute
for column in test_data.columns:
    if column == 'ClickedOnAd':
        continue
    column_data = test_data[column]
    print("Column: ", column)
    unique_values_dict[column] = {}
    for unique_value in column_data.unique():
        #checking the number of true values for each unique value
        true_values = len(test_data[(test_data[column] == unique_value) & (test_data['ClickedOnAd'] == 1)])
        unique_values_dict[column][unique_value] = true_values
        print("Unique value: ", unique_value, "True values: ", true_values ,'\n')

# Sort the unique values of each attribute based on the number of true values
for column, unique_values in unique_values_dict.items():
    sorted_unique_values = sorted(unique_values.items(), key=lambda x: x[1], reverse=True)
    test_data[column] = test_data[column].apply(lambda x: sorted_unique_values.index((x, unique_values[x])) if x in unique_values else 0)

# Sort the testing dataset based on the sorted attributes and unique values
test_data = test_data.reindex(columns=[column[0] for column in sorted_attributes])
test_data = test_data.sort_values(by=[column[0] for column in sorted_attributes])

# Write the sorted test data to a new csv file
test_data.to_csv('sorted_test_data.csv', index=False)
