from sklearn.datasets import make_circles, make_moons
import numpy as np
import matplotlib.pyplot as plt
X, Y = make_circles(random_state=42)
n = X.shape[0]
d = X.shape[1]


def perceptron_one_step(x, y, w):
    if y*(np.dot(x,w)) <= 0:
        return w + y*x
    return w

w = np.zeros(d+1)
for i in range(100):
    x = np.concatenate([X[i].flatten(), [1.0]])
    y = Y[i]
    w = perceptron_one_step(x, y, w)


w1, w2, b = w
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1],c=Y,s=10.0,cmap="jet")
x_grid = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 200)
ax.plot(x_grid, -w1/w2*x_grid - b, lw=3.0, label="learned hyperplane")
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.set_xlabel(R"$x^{(1)}$", fontsize=24)
ax.set_ylabel(R"$x^{(2)}$", fontsize=24)
plt.tight_layout()
plt.show()


# Exercise: show that the perceptron converges for a linearly separable dataset. Verify theorem on speed of convergence
#2. for the make_circles dataset, use a kernel perceptron to find the best separating hyperplane.
