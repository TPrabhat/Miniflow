"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import Miniflow as mini
import numpy as np

# Create 3 input objects
x, y, z = mini.Input(), mini.Input(), mini.Input()

# Create 3 Input nodes
inputs, weights, bias = mini.Input(), mini.Input(), mini.Input()
X, W, b = mini.Input(), mini.Input(), mini.Input()

# inputs for add node
f = mini.Add(x, y, z)

# inputs for multiply node
p = mini.Mul(x, y, z)
linear = mini.Linear(inputs, weights, bias)

# inputs for mean squared error node
pred_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

pred, a = mini.Input(), mini.Input()

feed_dict = {x: 4, y: 5, z: 10}

linear_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

# dictionary for MSE
MSE_dict = {pred: pred_, a: a_}

l = mini.Linear(X, W, b)
s = mini.Sigmoid(l)
cost = mini.MSE(pred, a)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

sigmoid_dict = {X: X_, W: W_, b: b_}

# Kahn's algorithm
graph = mini.topological_sort(feed_dict)
linear_graph = mini.topological_sort(linear_dict)
sigmoid_graph = mini.topological_sort(sigmoid_dict)
MSE_graph = mini.topological_sort(MSE_dict)

# forward pass
sum = mini.forward_pass(f, graph)
product = mini.forward_pass(p, graph)
linear_output = mini.forward_pass(linear, linear_graph)
sigmoid = mini.forward_pass(s, sigmoid_graph)
cost = mini.forward_pass(cost, MSE_graph)


# print outputs
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], sum))
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], product))
print(" Output of Linear pass {}".format(linear_output))
print(" Output of Sigmoid pass ", sigmoid)
print("Output of cost function", cost)