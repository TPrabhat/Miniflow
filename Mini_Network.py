"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import Miniflow as mini

x, y= mini.Input(), mini.Input()

f = mini.Add(x, y)

feed_dict = {x: 4, y: 5}

graph = mini.topological_sort(feed_dict)
output = mini.forward_pass(f, graph)

# should output 19
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
