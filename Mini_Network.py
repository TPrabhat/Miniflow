"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import Miniflow as mini

# Create 3 input objects
x, y, z = mini.Input(), mini.Input(), mini.Input()

f = mini.Add(x, y, z)
p = mini.Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = mini.topological_sort(feed_dict)
sum = mini.forward_pass(f, graph)
product = mini.forward_pass(p, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], sum))
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], product))