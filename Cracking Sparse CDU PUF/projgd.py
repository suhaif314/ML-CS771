import numpy as np
import time


def hard_thresholding(z, S):
    indices = np.argsort(-np.abs(z))[:S]
    u = np.zeros_like(z)
    u[indices] = z[indices]
    return u


def my_fit(X, y, w_init, learning_rate, S, max_iterations, tol):
    n, D = X.shape

    w = w_init
    t = 0

    while t < max_iterations:
        error = np.dot(X, w) - y
        z = w - learning_rate * np.dot(X.T, error) / n
        w_new = hard_thresholding(z, S)

        # Correction: Solve least squares problem using non-zero indices
        I = np.where(w_new != 0)[0]
        X_I = X[:, I]
        w_I = np.linalg.lstsq(X_I, y, rcond=None)[0]

        # Update w_new using the solution of the least squares problem
        w_new[I] = w_I

        if np.linalg.norm(w_new - w) < tol:
            print(t)
            break

        w = w_new
        t += 1

    return w


# Load challenge and response data from .dat files (assuming they are in a specific format)
challenge_data = np.loadtxt('train_challenges.dat')
response_data = np.loadtxt('train_responses.dat')

# Split the data into training and testing sets
train_X = challenge_data[:1280, :]  # Training challenge features
train_y = response_data[:1280]  # Training response targets
test_X = challenge_data[1280:, :]  # Testing challenge features
test_y = response_data[1280:]  # Testing response targets

# Initialize parameters
D = train_X.shape[1]  # Dimensionality of feature vectors
S = 512  # Sparsity level

n = challenge_data.shape[0]
X_transpose = challenge_data.T
w_init = np.linalg.lstsq(train_X, train_y, rcond=None)[0]

learning_rate = 4
max_iterations = 50
tol = 1e-4
n = train_X.shape[0]

# Run gradient descent with hard thresholding on the training data
tic = time.perf_counter()
w_optimal = my_fit(train_X, train_y, w_init, learning_rate, S, max_iterations, tol)
toc = time.perf_counter()

# Use the optimized weight vector for predictions on the test data
predictions = np.dot(test_X, w_optimal.T)

# Calculate mean absolute error on the test data
mae = np.mean(np.abs(predictions - test_y))
print("Mean Absolute Error:", mae)
print("Time:", toc - tic)
