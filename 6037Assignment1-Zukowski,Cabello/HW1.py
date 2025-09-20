import random
import math
# Arnaud Cabello - Noah Zukowski

# All team members have contributed in equal measure to this effort


# splits the data into the 3 categories: training, validation, and testing sets for analysis
def split_data(data, trainPercent, valPercent):
    random.shuffle(data)
    totalSize = len(data)
    trainSize = int(trainPercent * totalSize)
    valSize = int(valPercent * totalSize)
    
    train_data = data[:trainSize]
    valid_data = data[trainSize:trainSize + valSize]
    test_data = data[trainSize + valSize:]
    
    return train_data, valid_data, test_data

# CSV file data loading method
# skips the header and then returns a dataList (list of lists), data formating
def load_data(filepath):
    dataList = []
    with open(filepath, 'r') as file:
        for line in file.readlines()[1:]:  # Skipping the header
            features = list(map(float, line.strip().split(';'))) #split / format
            dataList.append(features)
    return dataList

# Change dataset into features (split1) / target (split2)
# 2 lists tuple, split1 all columns but last, split2 last column (target)
def split_X_Y(data):
    # Adding the bias term to be used later
    split1 = [row[:-1] + [1] for row in data]  
    split2 = [row[-1] for row in data]
    return split1, split2

# Mean Squared Error (MSE) calculation
# Val is the average of squared differences between predicitons and actual vals
def mse(num1, num2):
    return sum((num_1 - num_2)**2 for num_1, num_2 in zip(num1, num2)) / len(num1)

# computes the dot product with vector/weight
def dot_product(vector, weights):
    return sum(vec * weight for vec, weight in zip(vector, weights))

# Compute the different gradients of MSE
# takes into account the different respective weights
def compute_gradient(inputs, targets, weights):
    num_samples = len(targets)          # how many data points
    num_features = len(inputs[0])       # number of features
    
    gradients = [0] * num_features      
    
    # Loop through each data point
    for i in range(num_samples):
        predicted_value = dot_product(inputs[i], weights)
        error = predicted_value - targets[i]
        
        # Update the gradient for each feature
        for j in range(num_features):
            gradients[j] += (2 / num_samples) * error * inputs[i][j]
    
    return gradients

# standardizes the features in the initial data
def standardize_features(X):
    featureNum = len(X[0]) - 1  # ignore bias terms
    means = [0] * featureNum
    standardDev = [0] * featureNum

    # subtract mean and divid by standard deviation
    # also excludes bias col
    
    # Calculate means
    for i in range(featureNum):
        means[i] = sum(x[i] for x in X) / len(X)
    
    #calculate standard deviation
    for i in range(featureNum):
        standardDev[i] = math.sqrt(sum((x[i] - means[i])**2 for x in X) / len(X))
    
    # Standardize
    for row in X:
        for i in range(featureNum):
            if standardDev[i] != 0:
                row[i] = (row[i] - means[i]) / standardDev[i]
    
    return X

#Performs batch gradient descent without regularization.
def gradient_descent(inputs, targets, learning_rate=0.001, num_epochs=1000):
    num_features = len(inputs[0])  
    # initialize weights randomly for each feature (including bias)
    weights = [random.random() for _ in range(num_features)]  

    for epoch in range(num_epochs):
        gradients = compute_gradient(inputs, targets, weights)
        weights = [w - learning_rate * g for w, g in zip(weights, gradients)]
        
        # make predictions with the updated weights
        predictions = [dot_product(inputs[i], weights) for i in range(len(inputs))]
        
        # print progress (epoch number and current error)
        print(f"Epoch {epoch+1}, MSE: {mse(targets, predictions)}")
    
    return weights

# L2 regularization - Batch Gradient Descent  
def gradient_descent_L2(inputs, targets, regularization_strength=0.01, learning_rate=0.001, num_epochs=1000):
    num_features = len(inputs[0])  
    weights = [random.random() for _ in range(num_features)]
    
    # repeat for the given number of epochs
    for epoch in range(num_epochs):
        gradients = compute_gradient(inputs, targets, weights)
        l2_penalty = [2 * regularization_strength * w for w in weights]
        weights = [w - learning_rate * (g + p) for w, g, p in zip(weights, gradients, l2_penalty)]
        predictions = [dot_product(inputs[i], weights) for i in range(len(inputs))]
        
        # print progress (epoch number and current error)
        print(f"Epoch {epoch+1}, MSE: {mse(targets, predictions)}")
    
    return weights

# L1 regularization - Batch Gradient Descent
def sign(x):
    return math.copysign(1, x) if x != 0 else 0

# Perform gradient descent with L1 regularization (Lasso Regression)
def gradient_descent_L1(inputs, targets, regularization_strength=0.01, learning_rate=0.001, num_epochs=1000):
    num_features = len(inputs[0])
    # initialize weights randomly for each feature (including bias)
    weights = [random.random() for _ in range(num_features)]  
    
    # repeat for the number of epochs
    for epoch in range(num_epochs):
        gradients = compute_gradient(inputs, targets, weights)
        l1_penalty = [regularization_strength * sign(w) for w in weights]
        weights = [w - learning_rate * (g + p) for w, g, p in zip(weights, gradients, l1_penalty)]
        predictions = [dot_product(inputs[i], weights) for i in range(len(inputs))]
        print(f"Epoch {epoch+1}, MSE: {mse(targets, predictions)}")
    
    return weights

# Mini-batch Gradient Descent (no regu)
def mini_batch_gradient_descent(features, targets, batch_size=32, learning_rate=0.001, epochs=1000):
    if not features or not targets:
        raise ValueError("Features and targets cannot be empty.")
    
    num_features = len(features[0])
    theta = [random.random() for _ in range(num_features)]  # Initialize weights
    
    for epoch in range(epochs):
        # Shuffle the dataset
        dataset = list(zip(features, targets))
        random.shuffle(dataset)
        shuffled_features, shuffled_targets = zip(*dataset)
        
        # Process mini-batches
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            batch_X = shuffled_features[start:end]
            batch_Y = shuffled_targets[start:end]
            
            if not batch_X:
                continue  # Skip empty batch
            
            grads = compute_gradient(batch_X, batch_Y, theta)
            theta = [w - learning_rate * g for w, g in zip(theta, grads)]
        
        # Compute predictions and MSE for the epoch
        predictions = [dot_product(f, theta) for f in features]
        epoch_mse = mse(targets, predictions)
        print(f'Epoch {epoch+1}, MSE: {epoch_mse}')
    
    return theta

# L2 regularization - Mini-batch Gradient Descent
def mini_batch_gd_L2(X, Y, batch_sz=32, reg_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(X[0])
    w = [random.random() for _ in range(num_features)]
    
    for epoch in range(1, num_epochs + 1):
        # Shuffle data
        dataset = list(zip(X, Y))
        random.shuffle(dataset)
        X_shuff, Y_shuff = zip(*dataset)
        
        # Process mini-batches
        for start in range(0, len(X), batch_sz):
            end = start + batch_sz
            X_batch = X_shuff[start:end]
            Y_batch = Y_shuff[start:end]
            
            grad = compute_gradient(X_batch, Y_batch, w)
            l2_term = [2 * reg_strength * weight for weight in w]
            w = [weight - lr * (g + l2) for weight, g, l2 in zip(w, grad, l2_term)]
        
        # Compute predictions and MSE
        Y_pred = [dot_product(x_i, w) for x_i in X]
        print(f"Mini L2 Epoch {epoch}, MSE: {mse(Y, Y_pred)})")
    
    return w

# L1 regularization - Mini-batch Gradient Descent
def mini_batch_gd_L1(X, Y, batch_sz=32, reg_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(X[0])
    w = [random.random() for _ in range(num_features)]
    
    #shuffling data from epoch
    for epoch in range(1, num_epochs + 1):
        dataset = list(zip(X, Y))
        random.shuffle(dataset)
        X_shuff, Y_shuff = zip(*dataset)
        
        #process for each batch
        for start in range(0, len(X), batch_sz):
            end = start + batch_sz
            X_batch = X_shuff[start:end]
            Y_batch = Y_shuff[start:end]
            
            #gradients, regu L1 term
            grad = compute_gradient(X_batch, Y_batch, w)
            l1_term = [reg_strength * sign(weight) for weight in w]
            w = [weight - lr * (g + l1) for weight, g, l1 in zip(w, grad, l1_term)]
        
        #compute predictions
        Y_pred = [dot_product(x_i, w) for x_i in X]
        print(f"Mini L1 Epoch {epoch}, MSE: {mse(Y, Y_pred)})")
    
    return w

# Example usage using the winequality white dataset (loads dataset trains / splits given params and the seperates)
data = load_data('winequality-white.csv')
train_set, val_set, test_set = split_data(data, 0.7, 0.15)
X_train, Y_train = split_X_Y(train_set)

# Normalize features
X_train = standardize_features(X_train)

# Train using different the methods as provided above
weights_batch = gradient_descent(X_train, Y_train)
weights_l2 = gradient_descent_L2(X_train, Y_train)
weights_l1 = gradient_descent_L1(X_train, Y_train)
weights_mini_batch = mini_batch_gradient_descent(X_train, Y_train)
weights_mini_batch_l2 = mini_batch_gd_L2(X_train, Y_train)
weights_mini_batch_l1 = mini_batch_gd_L1(X_train, Y_train)