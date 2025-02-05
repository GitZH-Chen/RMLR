import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat = scipy.io.loadmat('spd_hyperplan.mat')
dotsize = 1
idx = [0, 2, 2, 2, 2]
fontsize = 20
fontsize_other = 18
num = 30
max_bound = 3

# Extract the cell arrays for different metrics
spd_hyperplan = mat['spd_hyperplan']

def extract_spd_points(metric_data, ith):
    X = metric_data[idx[ith], 0]
    Y = metric_data[idx[ith], 1]
    Z = metric_data[idx[ith], 2]
    return X, Y, Z

# Extract SPD points for different metrics
metrics = ['LEM', 'AIM', 'EM', 'BWM', 'LCM']
colors = ['r', 'g', 'b', 'm', 'c']  # Different colors for different metrics
spd_points = {}

for ith, metric in enumerate(metrics):
    metric_data = spd_hyperplan[metric][0, 0]
    spd_points[metric] = extract_spd_points(metric_data, ith)

# Generate boundary points
tmp_X, tmp_Z = np.meshgrid(np.linspace(0, max_bound, num), np.linspace(0, max_bound, num))
tmp_Y = np.sqrt(tmp_X * tmp_Z)
boundary_X = np.concatenate((tmp_X, tmp_X), axis=1)
boundary_Z = np.concatenate((tmp_Z, tmp_Z), axis=1)
boundary_Y = np.concatenate((tmp_Y, -tmp_Y), axis=1)

# Create subplots
fig = plt.figure(figsize=(20, 5))

for i, metric in enumerate(metrics):
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')

    # Plot SPD boundary points as a transparent scatter
    ax.scatter(boundary_X, boundary_Y, boundary_Z, s=dotsize, c='black')

    # Plot SPD points for the current metric with smaller dots
    X, Y, Z = spd_points[metric]
    ax.scatter(X, Y, Z, s=dotsize, c=colors[i], marker='.')

    # Set labels
    ax.set_xlabel('$x$', fontsize=fontsize_other)
    ax.set_ylabel('$y$', fontsize=fontsize_other)
    ax.set_zlabel('$z$', fontsize=fontsize_other, labelpad=15)  # Adjust label padding

    if metric == 'EM':
        metric = 'PEM'
    ax.set_title(f'{metric} SPD Hyperplane', fontsize=fontsize)

    # Set axis ticks with step = 1
    ax.set_xticks(np.arange(0, max_bound + 1, 1))
    ax.set_yticks(np.arange(-max_bound, max_bound + 1, 2))
    ax.set_zticks(np.arange(0, max_bound + 1, 1))

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=fontsize_other)

    # Set view angle
    ax.view_init(elev=30, azim=-60)

# Adjust layout to prevent overlap and give more room for titles
plt.tight_layout(pad=2.0)

# Adjust space specifically for z-axis labels if necessary
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save plot
# plt.savefig('hyperplane_spd_all.pdf', format='pdf', dpi=300)

# Show plot
plt.show()
