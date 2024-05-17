import numpy as np
import time

# 7.71 mae at lamda = 41

lamda = 41
max_iterations = 3000
S = 512


def hard_thresholding(z):
    indices = np.argsort(-np.abs(z))[:S]
    u = np.zeros_like(z)
    u[indices] = z[indices]
    return u


def soft_thresholding(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0.)


def subgradient(X, y, y_hat, w):
    ols_term = X.T.dot(y_hat - y) / y.size
    relaxation_term = lamda * np.sign(w)
    return ols_term + relaxation_term


def ista(X, y, w_0):
    w = w_0
    L = np.linalg.norm(X, ord=2) ** 2

    for i in range(max_iterations):
        y_predicted = np.dot(X, w)
        error = y - y_predicted
        loss = 1 / y.size * np.dot(error.T, error)
        # if i % 100 == 0:
        #     print(f"Iteration {i}: Loss: {loss}")

        u = w + np.dot(X.T, error) / L
        w = soft_thresholding(u, lamda / L)

    return w


def fista(X, y, w_0):
    w = w_0
    L = np.linalg.norm(X, ord=2) ** 2
    w_prev = w.copy()

    for i in range(max_iterations):
        y_predicted = np.dot(X, w)
        error = y - y_predicted
        loss = 1 / y.size * np.dot(error.T, error)
        # if i % 100 == 0:
        #     print(f"Iteration {i}: Loss: {loss}")

        m = w + ((i - 1.) / (i + 2.)) * (w - w_prev)
        u = m + np.dot(X.T, error) / L
        w_prev = w.copy()
        w = soft_thresholding(u, lamda/L)

    return w


X_trn = np.loadtxt("train_challenges.dat")
y_trn = np.loadtxt("train_responses.dat")
X_tst = X_trn[1280:]
y_tst = y_trn[1280:]
X_trn = X_trn[:1280]
y_trn = y_trn[:1280]
w_hat = np.array(np.linalg.lstsq(X_trn, y_trn, rcond=None)[0])
# w_hat = np.random.randn(2048)

tic = time.perf_counter()
# w_optimal = ista(X_trn, y_trn, w_hat)
w_optimal = fista(X_trn, y_trn, w_hat)
w_optimal = hard_thresholding(w_optimal)
toc = time.perf_counter()
mae_err = np.mean(np.abs((y_tst - np.matmul(X_tst, w_optimal))))
print("MAE error:", mae_err)
print("Time", toc-tic)
