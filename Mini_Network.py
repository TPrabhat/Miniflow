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

f = mini.Add(x, y, z)
p = mini.Mul(x, y, z)
linear = mini.Linear(inputs, weights, bias)

feed_dict = {x: 4, y: 5, z: 10}

linear_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

X, W, b = mini.Input(), mini.Input(), mini.Input()

f = mini.Linear(X, W, b)
g = mini.Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

sigmoid_dict = {X: X_, W: W_, b: b_}

graph = mini.topological_sort(feed_dict)
linear_graph = mini.topological_sort(linear_dict)
sigmoid_graph = mini.topological_sort(sigmoid_dict)

sum = mini.forward_pass(f, graph)
product = mini.forward_pass(p, graph)
linear_output = mini.forward_pass(linear, linear_graph)
sigmoid = mini.forward_pass(g, sigmoid_graph)


# print outputs
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], sum))
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], product))
print(" Output of Linear pass {}".format(linear_output))
print(" Output of Sigmoid pass {}".format(sigmoid))