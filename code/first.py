import numpy as np
import math

input_dimension = 728  # the pixel of picture
output_dimension = 10  # 0~9


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()


dimension = [input_dimension, output_dimension]
activation = [tanh, softmax]  # define activation function
distribution = [
    {'b': [0, 0]},  # first layer
    {'b': [0, 0],  # second layer
     'w': [-math.sqrt(6 / (input_dimension + output_dimension)), math.sqrt(6 / (input_dimension + output_dimension))]}
]


def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimension[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimension[layer-1], dimension[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters():
    parameters = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameters.append(layer_parameter)
    return parameters


parameters = init_parameters()
print(parameters)


def predict(image, parameters):
    l0_input = image + parameters[0]['b']
    l0_output = activation[0](l0_input)
    l1_input = np.dot(l0_output, parameters[1]['w']) + parameters[1]['b']
    l1_output = activation[1](l1_input)
    return l1_output


rand = np.random.rand(728)
print(predict(rand, parameters))
