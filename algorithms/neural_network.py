import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:

    def __init__(self, layer_sizes, lam=0.0, alpha=0.1, max_iters=1000, seed=None):
        self.layer_sizes = layer_sizes
        self.lam = lam
        self.alpha = alpha
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            rows = self.layer_sizes[i + 1]
            cols = self.layer_sizes[i] + 1  # +1 for bias
            self.weights.append(self.rng.uniform(-1.0, 1.0, size=(rows, cols)))

    def set_weights(self, weight_list):
        self.weights = [np.array(W, dtype=float) for W in weight_list]

    def forward_pass(self, x):
        # returns activations and z values for each layer
        activations = []
        pre_activations = [None]  # no z for input layer

        a = np.concatenate(([1.0], x))
        activations.append(a)

        for i, W in enumerate(self.weights):
            z = W @ a
            pre_activations.append(z)
            activated = sigmoid(z)
            if i == len(self.weights) - 1:
                a = activated
            else:
                a = np.concatenate(([1.0], activated))
            activations.append(a)

        return activations, pre_activations

    def compute_cost(self, X, Y):
        n = X.shape[0]
        total = 0.0
        for x, y in zip(X, Y):
            activations, _ = self.forward_pass(x)
            output = np.clip(activations[-1], 1e-12, 1.0 - 1e-12)
            total += np.sum(-y * np.log(output) - (1.0 - y) * np.log(1.0 - output))

        reg = sum(np.sum(W[:, 1:] ** 2) for W in self.weights)
        return total / n + (self.lam / (2.0 * n)) * reg

    def _compute_gradients(self, X, Y):
        n = X.shape[0]
        num_layers = len(self.weights)

        # accumulate gradients over all training examples
        grad_accum = [np.zeros_like(W) for W in self.weights]

        for x, y in zip(X, Y):
            activations, _ = self.forward_pass(x)

            delta = [None] * (num_layers + 1)
            delta[num_layers] = activations[-1] - y

            for k in range(num_layers - 1, 0, -1):
                hidden_a = activations[k][1:]  # strip bias
                sig_deriv = hidden_a * (1.0 - hidden_a)
                propagated = self.weights[k][:, 1:].T @ delta[k + 1]
                delta[k] = propagated * sig_deriv

            for k in range(num_layers):
                grad_accum[k] += np.outer(delta[k + 1], activations[k])

        # average and add regularization (no regularization on bias)
        reg_grads = []
        for k, W in enumerate(self.weights):
            avg = grad_accum[k] / n
            reg = (self.lam / n) * W.copy()
            reg[:, 0] = 0.0
            reg_grads.append(avg + reg)

        return reg_grads

    def fit(self, X, Y):
        cost_history = []
        for _ in range(self.max_iters):
            grads = self._compute_gradients(X, Y)
            for k in range(len(self.weights)):
                self.weights[k] -= self.alpha * grads[k]
            cost_history.append(self.compute_cost(X, Y))
        return cost_history

    def predict_proba(self, X):
        outputs = []
        for x in X:
            activations, _ = self.forward_pass(x)
            outputs.append(activations[-1])
        return np.array(outputs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)