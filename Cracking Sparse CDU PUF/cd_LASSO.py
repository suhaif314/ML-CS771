import numpy as np
import time


def soft_threshold(rho, lamda):
    """Soft threshold function used for normalized data and lasso regression"""
    if rho < -lamda:
        return rho + lamda
    elif rho > lamda:
        return rho - lamda
    else:
        return 0


def coordinate_descent_lasso(theta, X, y, lamda=.01, num_iters=100):
    """Coordinate gradient descent for lasso regression - for normalized data.
    The intercept parameter allows to specify whether we regularize theta_0"""

    # Initialisation of useful values
    m, n = X.shape
    X = X / (np.linalg.norm(X, axis=0))  # normalizing X in case it was not done before

    # Looping until max number of iterations
    for i in range(num_iters):

        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            X_j = X[:, j].reshape(-1, 1)
            y_predicted = X @ theta
            error = y - y_predicted
            loss = 1 / y.size * np.dot(error.T, error)
            if j % 100 == 0:
                print(f"Iteration {i}-{j}: Loss: {loss}")
            rho = X_j.T @ (error.reshape(-1, 1) + theta[j] * X_j)
            theta[j] = soft_threshold(rho, lamda)

    return theta.flatten()


X_trn = np.loadtxt("train_challenges.dat")
y_trn = np.loadtxt("train_responses.dat")
X_tst = X_trn[1280:]
y_tst = y_trn[1280:]
X_trn = X_trn[:1280]
y_trn = y_trn[:1280]
w_hat = np.array(np.linalg.lstsq(X_trn, y_trn, rcond=None)[0])
# w_hat = np.random.randn(2048)
tic = time.perf_counter()
w_optimal = coordinate_descent_lasso(w_hat, X_trn, y_trn)
# w_optimal = hard_thresholding(w_optimal)
toc = time.perf_counter()
mae_err = np.mean(np.abs((y_tst - np.matmul(X_tst, w_optimal))))
print("MAE error:", mae_err)
print("Time", toc-tic)
