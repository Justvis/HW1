import random
import math
# Arnaud Cabello - Noah Zukowski
# All team members have contributed in equal measure to this effort


# splits the data into the 3 categories: training, validation, and testing sets for analysis
def dataSplit(data, trainPercent, valPercent):
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
def load_data(givenPath):
    dataList = []
    #open file given the path
    with open(givenPath, 'r') as file:
        for l in file.readlines()[1:]:  # Skipping the header
            f = list(map(float, l.strip().split(';'))) #split / format
            dataList.append(f)
    return dataList

# Change dataset into features (split1) / target (split2)
# 2 lists tuple, split1 all columns but last, split2 last column (target)
def splitting(features):
    # Adding the bias term to be used later
    split1 = [row[:-1] + [1] for row in features]  
    #split 2, last col
    split2 = [row[-1] for row in features]
    return split1, split2

# Mean Squared Error (MSE) calculation
# Val is the average of squared differences between predicitons and actual vals
def meanSquaredErrorCalc(num1, num2):
    return sum((num_1 - num_2)**2 for num_1, num_2 in zip(num1, num2)) / len(num1)

# computes the dot product with vector/weight
def dotProd(vector, weights):
    return sum(vec * weight for vec, weight in zip(vector, weights))

# Compute the different gradients of MSE
# takes into account the different respective weights
def gradientMSECalc(inputs, targets, weights):
    num_samples = len(targets)          # how many data points
    num_features = len(inputs[0])       # number of features
    
    result = [0] * num_features      
    
    # Loop through each data point
    for n in range(num_samples):
        predicted_value = dotProd(inputs[n], weights)
        error = predicted_value - targets[n]
        
        # Update the gradient for each feature
        for i in range(num_features):
            result[i] += (2 / num_samples) * error * inputs[n][i]
    
    return result

# standardizes the features in the initial data
def standardizing(initial):
    featureNum = len(initial[0]) - 1  # ignore bias terms
    means = [0] * featureNum
    standardDev = [0] * featureNum

    # subtract mean and divid by standard deviation
    # also excludes bias col
    
    # Calculate means loop
    for n in range(featureNum):
        means[n] = sum(x[n] for x in initial) / len(initial)
    
    #calculate standard deviation loop
    for n in range(featureNum):
        standardDev[n] = math.sqrt(sum((x[n] - means[n])**2 for x in initial) / len(initial))
    
    # Standardize loop
    for row in initial:
        for n in range(featureNum):
            if standardDev[n] != 0:
                row[n] = (row[n] - means[n]) / standardDev[n]
    
    return initial

# L2 regularization method, mini grad
def miniBGradL2(X, Y, num_batch=32, reg_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(X[0])
    w = [random.random() for _ in range(num_features)]
    
    for epoch in range(1, num_epochs + 1):
        # Shuffle data
        dataset = list(zip(X, Y))
        random.shuffle(dataset)
        X_shuff, Y_shuff = zip(*dataset)
        
        # Process mini-batches
        for start in range(0, len(X), num_batch):
            end = start + num_batch
            X_batch = X_shuff[start:end]
            Y_batch = Y_shuff[start:end]
            
            grad = gradientMSECalc(X_batch, Y_batch, w)
            l2_term = [2 * reg_strength * weight for weight in w]
            w = [weight - lr * (g + l2) for weight, g, l2 in zip(w, grad, l2_term)]
        
        # Compute predictions and MSE
        Y_pred = [dotProd(x_i, w) for x_i in X]
        print(f"Mini L2 Epoch {epoch}, MSE: {meanSquaredErrorCalc(Y, Y_pred)})")
    
    return w

def miniBGradL1(X, Y, num_batch=32, reg_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(X[0])
    w = [random.random() for _ in range(num_features)]
    
    #shuffling data from epoch
    for epoch in range(1, num_epochs + 1):
        dataset = list(zip(X, Y))
        random.shuffle(dataset)
        X_shuff, Y_shuff = zip(*dataset)
        
        #process for each batch
        for start in range(0, len(X), num_batch):
            end = start + num_batch
            X_batch = X_shuff[start:end]
            Y_batch = Y_shuff[start:end]
            
            #gradients, regu L1 term
            grad = gradientMSECalc(X_batch, Y_batch, w)
            l1_term = [reg_strength * helperSign(weight) for weight in w]
            w = [weight - lr * (g + l1) for weight, g, l1 in zip(w, grad, l1_term)]
        
        #compute predictions
        Y_pred = [dotProd(x_i, w) for x_i in X]
        print(f"Mini L1 Epoch {epoch}, MSE: {meanSquaredErrorCalc(Y, Y_pred)})")
    
    return w

#Performs batch gradient descent without regularization.
def gradDesc(inputs, targets, lr=0.001, num_epochs=1000):
    num_features = len(inputs[0])  
    # initialize weights randomly for each feature (including bias)
    we = [random.random() for _ in range(num_features)]  

    for epoch in range(num_epochs):
        #grad calculation for MSE
        grad = gradientMSECalc(inputs, targets, we)
        we = [w - lr * g for w, g in zip(we, grad)]
        
        # make predictions with the updated weights
        predictions = [dotProd(inputs[i], we) for i in range(len(inputs))]
        
        # print progress (epoch number and current error)
        print(f"Epoch grad descent: {epoch+1}, MSE: {meanSquaredErrorCalc(targets, predictions)}")
    
    return we

# batch grad desc (L2) 
def gradDescL2(inputs, targets, regularization_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(inputs[0])  
    we = [random.random() for _ in range(num_features)]
    
    # repeat for the given number of epochs
    for epoch in range(num_epochs):
        #MSE calc
        gradients = gradientMSECalc(inputs, targets, we)
        pen = [2 * regularization_strength * w for w in we]
        we = [weight - lr * (g + l2) for weight, g, l2 in zip(we, gradients, pen)]
        #dot prod calc
        predictions = [dotProd(inputs[n], we) for n in range(len(inputs))]
        
        # print progress (epoch number and current error)
        print(f"Epoch GDL2: {epoch+1}, MSE: {meanSquaredErrorCalc(targets, predictions)}")
    
    return we

# helper function for copysign
def helperSign(input):
    return math.copysign(1, input) if input != 0 else 0

# Perform gradient descent with L1 regularization (Lasso Regression)
def gradDescL1(inputs, targets, regularization_strength=0.01, lr=0.001, num_epochs=1000):
    num_features = len(inputs[0])
    # initialize weights randomly for each feature (including bias)
    we = [random.random() for _ in range(num_features)]  
    
    # repeat for the number of epochs
    for epoch in range(num_epochs):
        #MSE calc for grad
        gradients = gradientMSECalc(inputs, targets, we)
        pen = [regularization_strength * helperSign(w) for w in we]
        #weight calc
        we = [weight - lr * (g + l1) for weight, g, l1 in zip(we, gradients, pen)]
        predictions = [dotProd(inputs[n], we) for n in range(len(inputs))]
        print(f"Epoch L1 {epoch+1}, MSE: {meanSquaredErrorCalc(targets, predictions)}")
    
    return we

# Mini-batch Gradient Descent (no regu)
def miniBGradDesc(features, targets, num_batch=32, lr=0.001, num_epochs=1000):
    if not features or not targets:
        raise ValueError("features cant be empty")
    # Initialize weights, get theta
    num_features = len(features[0])
    theta = [random.random() for _ in range(num_features)]  
    
    for epoch in range(num_epochs):
        # Shuffle the dataset
        dataset = list(zip(features, targets))
        random.shuffle(dataset)
        shuffled_features, shuffled_targets = zip(*dataset)
        
        # Process mini-batches
        for start in range(0, len(features), num_batch):
            end = start + num_batch
            batch_X = shuffled_features[start:end]
            batch_Y = shuffled_targets[start:end]
            
            if not batch_X:
                continue  # Skip empty batch
            
            #calculations for theta and grads
            grads = gradientMSECalc(batch_X, batch_Y, theta)
            theta = [w - lr * g for w, g in zip(theta, grads)]
        
        # Compute predictions and MSE for the epoch
        predictions = [dotProd(f, theta) for f in features]
        epoch_mse = meanSquaredErrorCalc(targets, predictions)
        print(f'Epoch mini batch {epoch+1}, MSE: {epoch_mse}')
    
    return theta

# Example usage using the winequality white dataset (loads dataset trains / splits given params and the seperates)
data = load_data('winequality-white.csv')
#data splitting
train_data, val_data, test_data = dataSplit(data, 0.7, 0.15)
trainX, trainY = splitting(train_data)

# standardizing train X data
trainX = standardizing(trainX)

# Initial all different training methods and run them
weightMiniGradL2 = miniBGradL2(trainX, trainY)
weightMiniBGradL1 = miniBGradL1(trainX, trainY)
weightB = gradDesc(trainX, trainY)
weightL1 = gradDescL1(trainX, trainY)
weightL2 = gradDescL2(trainX, trainY)
weightMiniB = miniBGradDesc(trainX, trainY)
