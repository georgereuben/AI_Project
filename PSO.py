import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

import numpy as np

def pso_search(data, model, most_important_attr, iterations=150, num_particles=20, omega=0.9, phi_p=0.5, phi_g=0.5):
    # Sort the test data by the most important attribute
    sorted_data = data.sort_values(by=[most_important_attr])
    
    # Get the column order of the data
    column_order = sorted_data.columnscolumns
    
    # Get the number of dimensions
    num_dimensions = len(column_order)
    
    # Initialize the particles
    particles = []
    particle_best_solutions = []
    global_best_solution = None
    global_best_score = -np.inf
    
    for i in range(num_particles):
        # Initialize the particle's position to a random row in the data
        particle_position = sorted_data.sample(n=1).iloc[0].loc[column_order]
        
        # Evaluate the objective function on the particle's position
        particle_score = objective_function(particle_position, model)
        
        # Set the particle's best position to be its initial position
        particle_best_position = particle_position
        particle_best_score = particle_score
        
        # Update the global best solution if necessary
        if particle_score > global_best_score:
            global_best_solution = particle_position
            global_best_score = particle_score
        
        # Add the particle to the list of particles
        particles.append((particle_position, particle_best_position))
        particle_best_solutions.append(particle_best_position)
    
    # Iterate for the specified number of iterations
    for i in range(iterations):
        for j in range(num_particles):
            # Get the current particle's position and best position
            particle_position, particle_best_position = particles[j]
            
            # Generate a new velocity for the particle
            velocity = omega * np.array(particles[j][0] - particles[j][1])
            velocity += phi_p * np.random.rand(num_dimensions) * (particle_best_position - particle_position)
            velocity += phi_g * np.random.rand(num_dimensions) * (global_best_solution - particle_position)
            
            # Update the particle's position
            particle_position += velocity
            
            # Clip the particle's position to the valid range
            particle_position = np.clip(particle_position, sorted_data.iloc[0].loc[column_order], sorted_data.iloc[-1].loc[column_order])
            
            # Evaluate the objective function on the new position
            particle_score = objective_function(particle_position, model)
            
            # Update the particle's best position if necessary
            if particle_score > objective_function(particle_best_position, model):
                particle_best_position = particle_position
            
            # Update the global best solution if necessary
            if particle_score > global_best_score:
                global_best_solution = particle_position
                global_best_score = particle_score
            
            # Update the particle's position and best position in the list of particles
            particles[j] = (particle_position, particle_best_position)
            particle_best_solutions[j] = particle_best_position
        
        # Print the global best solution and score for debugging purposes
        print(f"Iteration {i+1}: {global_best_solution}\nScore: {global_best_score}")
    
    global foo
    foo = global_best_score
    # Return the best row found
    return global_best_solution

# Run hill climbing search on the most important attribute
most_important_attr = sorted_attributes[0][0]
best_row = pso_search(sorted_test_data, linear_model, most_important_attr)

# Print the best state
print("\n\n\n\Initial state: ", sorted_test_data.iloc[130])
print("\n\n\n\Best state: ", best_row, "Best Score", foo)