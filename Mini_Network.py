"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import Miniflow as mini

# Create 3 input objects
x, y, z = mini.Input(), mini.Input(), mini.Input()

# store the addition function in the variable f
f = mini.Add(x, y, z)

# initialize nodes with values
feed_dict = {x: 4, y: 5, z: 1}


graph = mini.topological_sort(feed_dict)
output = mini.forward_pass(f, graph)

# should output the sum
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
