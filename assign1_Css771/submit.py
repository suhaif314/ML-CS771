import numpy as np

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding

    w_init = np.linalg.lstsq(X_trn, y_trn, rcond=None)[0]
    learning_rate = 5
    max_iterations = 50
    tol = 1e-4
    n = X_trn.shape[0]
    w = w_init
    t = 0
    S = 512

    def hard_thresholding(z, S):
        indices = np.argsort(-np.abs(z))[:S]
        u = np.zeros_like(z)
        u[indices] = z[indices]
        return u

    while t < max_iterations:
        error = np.dot(X_trn, w) - y_trn
        z = w - learning_rate * np.dot(X_trn.T, error) / n
        w_new = hard_thresholding(z, S)

        # Correction: Solve least squares problem using non-zero indices
        I = np.where(w_new != 0)[0]
        X_I = X_trn[:, I]
        w_I = np.linalg.lstsq(X_I, y_trn, rcond=None)[0]

        # Update w_new using the solution of the least squares problem
        w_new[I] = w_I

        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new
        t += 1

    model = w

    return model					# Return the trained model

