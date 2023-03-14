import numpy as np
import random as rd


def generate_current_vector_w(vector_w, variability):
    current_vector_w = np.zeros(vector_w.size)
    for w in range(vector_w.size):
        mu, sigma = vector_w[w], vector_w[w] * variability
        new_factor = np.random.normal(mu, sigma, 1)
        current_vector_w[w] = new_factor
    return current_vector_w


def generate_dataset(number_training_examples, number_features, variability, b):
    """
    We begin by generating a range for us to generate random x-values with, as well as generating a vector for
    all of our w values, and by generating an empty dataset
    """

    ranges = np.round(np.random.uniform(1, 100, number_features))
    vector_w = np.round(np.random.uniform(0, 100, number_features), 1)
    data = np.zeros((number_training_examples, number_features))

    """ 
    We now generate the values for our training examples and assign them into our dataset
    """
    for i in range(number_training_examples):
        for j in range(number_features):
            data[i][j] = rd.uniform(0, ranges[j])

    """
    We now create our vector for our outputs
    """

    vector_y = np.zeros(data.size)

    for training_example in range(data.shape[0]):
        current_vector_w = generate_current_vector_w(vector_w, variability)
        y = np.dot(data[training_example], current_vector_w) + b
        vector_y[training_example] = y

    return data, vector_y, vector_w


def linear_regression_line(w, x, b):
    """
    Returns our predicted value of y
    """
    return np.dot(w, x) + b


def cost_function(w, b, data, out):
    """
    Calculates the cost of our current w and b
    """
    cost = 0
    shape = data.shape
    rows = shape[0]
    for i in range(rows):
        cost += (linear_regression_line(w, data[i], b) - out[i]) ** 2
    return cost / (2 * rows)


def gradient_descent_helper(weights, bias, data, output):
    data_rows, data_columns = data.shape

    new_bias = 0
    new_weights_vector = np.zeros(weights.size)

    for training_example in range(data_rows):
        error = linear_regression_line(weights, data[training_example], bias) - output[training_example]
        for feature in range(data_columns):
            new_weights_vector[feature] += error * data[training_example][feature]
        new_bias += error
    new_bias /= data_rows
    new_weights_vector /= data_rows

    return new_weights_vector, new_bias


def gradient_descent(weights, bias, learning_rate, data, output, iterations):
    for i in range(iterations):
        new_weights_vector, new_bias = gradient_descent_helper(weights, bias, data, output)

        weights -= learning_rate * new_weights_vector
        bias -= learning_rate * new_bias

        current_cost = cost_function(weights, bias, data, output)
        print(f"Cost: {current_cost}")
    return weights, bias


dataset, out, vector_w = generate_dataset(100, 5, 0.1, 15)
weight_vector, b = gradient_descent(np.zeros(5), 0, 5.0e-6, dataset, out, 25000)

print(f"Calculated Weight: {weight_vector},   Actual Weight: {vector_w}")
print(f"Calculated Bias: {b},  Actual Bias: 15")
print(linear_regression_line(weight_vector, dataset[0], b))
print(out[0])


"""
To decide number of iterations, stop iterating when in the last 10 iterations, the cost has barely changed by some percentage
"""