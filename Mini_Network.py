"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import Miniflow as mini

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

graph = mini.topological_sort(feed_dict)
linear_graph = mini.topological_sort(linear_dict)

sum = mini.forward_pass(f, graph)
product = mini.forward_pass(p, graph)
linear_output = mini.forward_pass(linear, linear_graph)

# print outputs
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], sum))
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], product))
print(" Output of Linear pass {}".format(linear_output))