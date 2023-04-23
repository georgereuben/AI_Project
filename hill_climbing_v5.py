import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = pd.read_csv('Cleaned_Advertisements.csv')
#dataset = dataset.drop(['Unnamed: 0'], axis=1)

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

# Identify the most important attribute
most_important_attr = sorted_attributes[0][0]

# Create groups of 10 based on the most important attribute
test_data['group'] = pd.cut(test_data[most_important_attr], bins=range(int(test_data[most_important_attr].min()), int(test_data[most_important_attr].max())+11, 10), labels=False)

# Calculate the number of true values of ClickedOnAd for each group and sort the groups
group_counts = test_data.groupby('group')['ClickedOnAd'].sum().sort_values(ascending=False).to_dict()

# Sort the testing dataset based on the sorted attributes and unique values
#test_data = test_data.reindex(columns=[column[0] for column in sorted_attributes])
test_data = test_data.sort_values(by=[column[0] for column in sorted_attributes])
test_data = test_data.drop(['ClickedOnAd'], axis=1)

# Create groups of 10 based on the most important attribute
test_data['group'] = pd.cut(test_data[most_important_attr], bins=range(int(test_data[most_important_attr].min()), int(test_data[most_important_attr].max())+11, 10), labels=False)

# Assign a rank to each group based on the number of true values of ClickedOnAd and sort the test data
test_data['rank'] = test_data['group'].apply(lambda x: sorted(list(group_counts.keys()), key=lambda y: -group_counts[y]).index(x)+1)
test_data = test_data.sort_values(['group', 'rank'])

# Write the sorted test data to a new csv file
test_data.to_csv('sorted_test_data.csv', index=False)

# Load the sorted test data
sorted_test_data = pd.read_csv('sorted_test_data.csv')
sorted_test_data.drop(['group', 'rank'], axis=1, inplace=True)
print("sorted test data columns : ", sorted_test_data.columns)

test_rows = list(range(len(sorted_test_data)))
test_state = sorted_test_data.iloc[test_rows[0]]
print("\n\n\n\n\ntest state : ", test_state)

# Load the linear model trained on the training data
linear_model = LogisticRegression()
y_train = train_data['ClickedOnAd']
X_train = train_data.drop(['ClickedOnAd'], axis=1)
linear_model.fit(X_train, y_train)

def predict_proba_on_row(row,model):
    X = row.to_numpy().reshape(1, -1)
    probas = model.predict_proba(X)
    return probas[0, 1]

def objective_function(row, model, column_order):
    # Create a DataFrame with the row data and reorder the columns
    row_df = pd.DataFrame(row, index=[0], columns=column_order)

    # Predict the probability of the row being clicked on using the predict_proba_on_row function
    proba = predict_proba_on_row(row_df, model)

    # Return the negative log-likelihood of the predicted probability
    return -np.log(proba)

def hill_climbing_search(data, model, most_important_attr, max_iterations=100):
    
    # Create a list of rows to work on
    rows = list(range(len(data)))

    # Define the column order for the row data
    column_order = train_data.drop(['ClickedOnAd'], axis=1).columns
    
    # Initialize the current state
    current_state = test_data.iloc[0].loc[column_order].values.reshape(1, -1)
    
    # Evaluate the current state
    current_score = objective_function(current_state, linear_model, column_order)
    
    # Iterate for a set number of iterations
    for i in range(max_iterations):
        
        # Get a random neighbour
        neighbour_index = random.choice([-3, -2, -1, 0, 1, 2, 3])
        neighbour_state = data.iloc[rows[rows.index(data.index.get_loc(current_state.name)) + neighbour_index]].loc[column_order].values
        
        # Evaluate the neighbour
        neighbour_score = objective_function(neighbour_state, data, model, most_important_attr)
        
        # Check if the neighbour is better than the current state
        if neighbour_score > current_score:
            current_state = neighbour_state
            current_score = neighbour_score
    
    return current_state, current_score

def get_neighbours(state, data):
    current_index = state.index[0]
    max_index = data.index.max()
    neighbours = []

    if current_index > 0:
        neighbours.append(data.loc[current_index - 1])
    if current_index < max_index:
        neighbours.append(data.loc[current_index + 1])

    return neighbours

# Run hill climbing search on the most important attribute
most_important_attr = sorted_attributes[0][0]
best_row = hill_climbing_search(sorted_test_data, linear_model, most_important_attr)

# Print the best state
print("\n\n\n\n\n\n\Best state: ", best_row[0])