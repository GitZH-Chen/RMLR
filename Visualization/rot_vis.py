import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat = scipy.io.loadmat('axi_angle.mat')

# Access the cell array and convert to Python lists
axi_angle = mat['axi_angle']
axi_angle_python = [axi_angle[0, i] for i in range(axi_angle.shape[1])]

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Show the grid with a light grey color and set the background color
pane_color = (0.9, 0.9, 0.9, 0.1)  # Lighter grey with more transparency
ax.w_xaxis.set_pane_color(pane_color)
ax.w_yaxis.set_pane_color(pane_color)
ax.w_zaxis.set_pane_color(pane_color)

# Show the grid with the same color and transparency as the panes
grid_color = (0.98, 0.98, 0.98, 0.1)
ax.grid(True, color=grid_color)

# Draw the boundary of the ball
n = 40
X, Y, Z = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n))
sphere_x, sphere_y, sphere_z = np.pi * np.sin(Y) * np.cos(X), np.pi * np.sin(Y) * np.sin(X), np.pi * np.cos(Y)

dotsize = 0.1  # Adjust the dot size as needed
ax.scatter(sphere_x, sphere_y, sphere_z, s=0.1, c='grey', marker='.',alpha=0.1)

# Plot SO(3) points
ax.scatter(axi_angle_python[0][0, :], axi_angle_python[0][1, :], axi_angle_python[0][2, :], s=dotsize, c='g', marker='.')

# Set labels with LaTeX and larger fontsize
fontsize = 16
ax.set_xlabel(r'$x$', fontsize=fontsize)
ax.set_ylabel(r'$y$', fontsize=fontsize)
ax.set_zlabel(r'$z$', fontsize=fontsize)

# Set axis limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])

# Set ticks with larger fontsize
ax.set_xticks(np.arange(-4, 4, 2))
ax.set_yticks(np.arange(-4, 4, 2))
ax.set_zticks(np.arange(-4, 4, 2))
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_title('Lie Hyperplane on Rotations',fontsize=fontsize)

# Adjust the view
# ax.view_init(elev=30, azim=-60)

# Save the figure
figWidth = 14  # width in inches
figHeight = 7  # height in inches
fig.set_size_inches(figWidth, figHeight)
# plt.savefig('hyperplane_rot.pdf', format='pdf', dpi=300)

plt.show()
