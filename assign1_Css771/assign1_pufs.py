import numpy as np

class SparseLinearModel:
    def __init__(self, D, S):
        self.D = D  # Total number of CDUs
        self.S = S  # Number of active CDUs
        self.w = np.zeros(D)  # Weight vector

    def my_fit(self, challenges, responses):
        X = np.zeros((len(challenges), self.D))
        for i, c in enumerate(challenges):
            X[i] = c
        
        y = np.array(responses)
        
        # Solve the sparse linear regression problem
        #self.w, _ = np.linalg.lstsq(X, y, rcond=None)
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]

        self.w[np.argsort(np.abs(self.w))[:-self.S]] = 0  # Apply hard thresholding to enforce sparsity
    
    def predict(self, challenge):
        return np.dot(challenge, self.w)

# Example usage:
model = SparseLinearModel(D=2048, S=512)

# Load training data
train_challenges = np.loadtxt('train_challenges.dat', delimiter=' ')
train_responses = np.loadtxt('train_responses.dat', delimiter=' ')

# Train the model
model.my_fit(train_challenges, train_responses)

# Load test data
test_challenges = np.loadtxt('dummy_test_challenges.dat', delimiter=' ')

# Make predictions on the test data
test_responses = [model.predict(c) for c in test_challenges]

# Print the predicted responses
for response in test_responses:
    print(response)
