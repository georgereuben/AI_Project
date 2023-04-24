import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

def debug_print(*args):
    print("\n\n",*args, flush=True)

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

foo = 0;            #to store the best score

# Create groups of 10 based on the most important attribute
test_data['group'] = pd.cut(test_data[most_important_attr], bins=range(int(test_data[most_important_attr].min()), int(test_data[most_important_attr].max())+11, 10), labels=False)

# Assign a rank to each group based on the number of true values of ClickedOnAd and sort the test data
test_data['rank'] = test_data['group'].apply(lambda x: sorted(list(group_counts.keys()), key=lambda y: -group_counts[y]).index(x)+1)
test_data = test_data.sort_values(['group', 'rank'])
debug_print(f"test data columns : {test_data.columns}")

# Write the sorted test data to a new csv file
test_data.to_csv('sorted_test_data.csv', index=False)

# Load the sorted test data
sorted_test_data = pd.read_csv('sorted_test_data.csv')
sorted_test_data.drop(['group','rank'], axis=1, inplace=True)
print("sorted test data columns : ", sorted_test_data.columns)

test_rows = list(range(len(sorted_test_data)))
test_state = sorted_test_data.iloc[test_rows[0]]
print("\n\n\n\n\ntest state : ", test_state)

# Load the linear model trained on the training data
y = train_data['ClickedOnAd']
X = train_data.drop(['ClickedOnAd'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)
debug_print(f"linear model true value probability prediction: {linear_model.predict_proba(sorted_test_data)}")

def objective_function(row, model):

    #convert row (pandas.core.series.Series) to dataframe (pandas.core.frame.DataFrame)
    row = row.to_frame().transpose()

    # Predict the probability of the row being clicked on using the predict_proba_on_row function
    proba = model.predict_proba(row)
    proba = proba[0][0]
    #debug_print(f"proba : {proba}")

    return float(proba)

def local_beam_search(data, model, most_important_attr, beam_width=5, iterations=50000):
    # Sort the test data by the most important attribute
    sorted_data = data.sort_values(by=[most_important_attr])
    
    # Get the column order of the data
    column_order = sorted_data.columns
    
    # Initialize the current states to be the top 'beam_width' rows of the sorted data
    current_states = sorted_data.iloc[:beam_width]
    debug_print(f"Current states: {current_states}")
    debug_print(f"Current states data type: {type(current_states)}")
    
    for i in range(iterations):
        new_states = []
        # Evaluate the objective function on the current states
        current_scores = current_states.apply(lambda x: objective_function(x, model), axis=1)
        
        # Search the neighbourhood for each current state
        for current_index, current_state in enumerate(current_states):
            # Get the index of the current state
            current_index = sorted_data.index.get_loc(current_state.name)
            
            # Get the indices of the neighbouring states
            if current_index == 0:
                neighbour_indices = [1]
            elif current_index == len(sorted_data) - 1:
                neighbour_indices = [-1]
            else:
                # Generate a list of neighbour indices with a Gaussian probability distribution centered at 0
                neighbour_indices = np.random.normal(loc=0, scale=1, size=20).round().astype(int)
                neighbour_indices = np.clip(neighbour_indices, -current_index, len(sorted_data) - current_index - 1)
            
            # Search the neighbourhood for each current state
            for neighbour_index in neighbour_indices:
                # Get the neighbour state
                neighbour_state = sorted_data.iloc[current_index + neighbour_index].loc[column_order]
                
                # Evaluate the objective function on the neighbour state
                neighbour_score = objective_function(neighbour_state, model)
                
                # Add the neighbour state to the new states list
                new_states.append(neighbour_state)
                
        # Choose the top 'beam_width' states with the best scores
        scores = [objective_function(state, model) for state in new_states]
        top_indices = np.argsort(scores)[:beam_width]
        current_states = [new_states[i] for i in top_indices]
        
        # Print the current states and scores for debugging purposes
        print(f"Iteration {i+1}: {current_states}\nScore: {scores[top_indices]}")
    
    global foo
    foo = scores[top_indices]
    # Return the best row found
    return current_states[0]

# Run search on the most important attribute
most_important_attr = sorted_attributes[0][0]
best_row = local_beam_search(sorted_test_data, linear_model, most_important_attr, beam_width=5, iterations=50000)

# Print the best state
print("\n\n\n\Initial state: ", sorted_test_data.iloc[130])
print("\n\n\n\Best state: ", foo)























































































