import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the hat function that is zero on the boundary
def hat_function(x, y):
    # The peak is at (0.5, 0.5), and it linearly decreases to 0 at the boundaries
    value = 1 - 2 * (np.abs(x - 0.5) + np.abs(y - 0.5))
    return np.maximum(value, 0)

# Create a grid of points over the unit square
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = hat_function(X, Y)

# Plot the hat function using a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('hat_function(x, y)')
ax.set_title('Finite Element Hat Function over Unit Square')

# Show the plot
plt.show()
