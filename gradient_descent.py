import numpy as np
import random as rd


def generate_current_vector_w(vector_w, variability):
    current_vector_w = []
    for w in vector_w:
        mu, sigma = w, w * variability
        new_factor = np.random.normal(mu, sigma, 1)
        current_vector_w.append(new_factor)
    return current_vector_w


def generate_dataset(number_training_examples, number_features, variability, b):
    """
    We begin by generating a range for us to generate random x-values with, as well as generating a vector for
    all of our w values, and by generating an empty dataset
    """

    ranges = np.round(np.random.uniform(1, 100, number_features))
    vector_w = np.round(np.random.uniform(0, 100, number_features), 1)
    data = np.empty((number_training_examples, number_features))

    """ 
    We now generate the values for our training examples and assign them into our dataset
    """
    for i in range(number_training_examples):
        for j in range(number_features):
            data[i][j] = rd.uniform(0, ranges[j])

    """
    We now create our vector for our outputs
    """

    vector_y = []

    for training_example in data:
        current_vector_w = generate_current_vector_w(vector_w, variability)
        y = np.dot(training_example, current_vector_w) + b
        vector_y.append(y)

    vector_y = np.array(vector_y)

    return data, vector_y


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
        cost = cost + (linear_regression_line(w, data[i], b) - out[i]) ** 2
    return cost / 2

def gradient_descent ():
    pass
