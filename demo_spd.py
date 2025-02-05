import torch as th
from RieNets.spdnets.SPDMLR import SPDRMLR

# Set parameters
batch_size = 8  # Batch size
n = 5  # Dimension of SPD matrices
c = 3  # Number of classes

# Generate random SPD matrices of shape (batch_size, n, n)
X = th.randn(batch_size, 1, n, n)
X = X @ X.transpose(-1, -2)  # Ensure positive definiteness
X += th.eye(n) * 1e-3  # Add a small perturbation to guarantee strict positive definiteness

# Initialize the model
model = SPDRMLR(n=n, c=c, metric='LEM')

# Forward computation
output = model(X)

# Print results
print("Input X shape:", X.shape)
print("Output shape:", output.shape)
print("Output:", output)
