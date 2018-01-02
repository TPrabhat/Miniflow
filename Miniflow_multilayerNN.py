import numpy as np
import Miniflow as mini

X, W, b = mini.Input(), mini.Input(), mini.Input()
y = mini.Input()
f = mini.Linear(X, W, b)
a = mini.Sigmoid(f)
cost = mini.MSE(y, a)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_ = np.array([1, 2])

feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
}

graph = mini.topological_sort(feed_dict)
mini.forward_and_backward(graph)
# return the gradients for each Input
gradients = [t.gradients[t] for t in [X, y, W, b]]

print(gradients)
