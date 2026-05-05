import numpy as np


#Gaussian Naive Bayes for EC4 
# our current Multinomial NB needs binning to handle numerical features which throws away info so here we modeled each feature as a gaussian per class and also extended to multiclass since the original was hardcoded to pos/neg

class GaussianNaiveBayes:

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

        self.classes_ = None
        self.priors_  = None
        self.means_   = None
        self.vars_    = None


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]
        n_total    = X.shape[0]

        self.means_  = np.zeros((n_classes, n_features))
        self.vars_   = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

    #epsilon sheilds against 0 variance 
        eps = self.var_smoothing * X.var(axis=0).max()

        i = 0
        while i < n_classes:
            cls   = self.classes_[i]
            X_c   = X[y == cls]

            self.priors_[i]  = X_c.shape[0] / n_total
            self.means_[i]   = X_c.mean(axis=0)
            self.vars_[i]    = X_c.var(axis=0) + eps

            i = i + 1

        return self


    def _log_likelihood(self, X):
#log P(x | class) for each class, shape -> (n_samples, n_classes)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        out = np.zeros((n_samples, n_classes))

        c = 0
        while c < n_classes:
            mu  = self.means_[c]
            var = self.vars_[c]

#log( N(x; mu, var) ) = -0.5 * sum( log(2*pi*var) + (x-mu)^2 / var )
            const_term = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            dist_term  = -0.5 * np.sum(((X - mu) ** 2) / var, axis=1)

            out[:, c] = const_term + dist_term
            c = c + 1

        return out


    def predict(self, X):
        X = np.asarray(X, dtype=float)

        log_lik   = self._log_likelihood(X)
        log_prior = np.log(self.priors_)

#posterior is proportional to likelihood * prior
#argmax doesn't need no normalization
        log_post  = log_lik + log_prior
        best_idx  = np.argmax(log_post, axis=1)

        return self.classes_[best_idx]


    def predict_log_proba(self, X):
#normalized the log-probabilities, in case we ever want to plug into an ensemble
        X = np.asarray(X, dtype=float)
        log_post = self._log_likelihood(X) + np.log(self.priors_)
        max_lp   = log_post.max(axis=1, keepdims=True)
        log_post = log_post - max_lp
        log_z    = np.log(np.exp(log_post).sum(axis=1, keepdims=True))
        return log_post - log_z