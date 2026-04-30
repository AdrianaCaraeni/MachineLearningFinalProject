from collections import defaultdict
import math


class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prior = {}
        self.word_counts = {}
        self.total_word_counts = {}
        self.vocab = set()

    def train(self, pos_train, neg_train, vocab):
        self.vocab = vocab
        total_docs = len(pos_train) + len(neg_train)
        training_data = {'positive': pos_train, 'negative': neg_train}

        for cls, docs in training_data.items():
            self.prior[cls] = len(docs) / total_docs
            counts = defaultdict(int)
            for doc in docs:
                for word in doc:
                    counts[word] += 1
            self.word_counts[cls] = counts
            self.total_word_counts[cls] = sum(counts.values())

    def get_word_prob(self, word, cls):
        n_wk = self.word_counts[cls].get(word, 0)
        n_total = self.total_word_counts[cls]
        vocab_size = len(self.vocab)
        return (n_wk + self.alpha) / (n_total + self.alpha * vocab_size)

    # Q1 - standard product formulation, no log
    def classify_standard(self, doc):
        probs = {}
        for cls in ['positive', 'negative']:
            prob = self.prior[cls]
            for word in doc:
                prob *= self.get_word_prob(word, cls)
            probs[cls] = prob
        return max(probs, key=lambda cls: probs[cls])

    # Q2+ - use log probs to avoid underflow
    def classify(self, doc):
        log_probs = {}
        for cls in ['positive', 'negative']:
            log_prob = math.log(self.prior[cls])
            for word in doc:
                log_prob += math.log(self.get_word_prob(word, cls))
            log_probs[cls] = log_prob
        return max(log_probs, key=lambda cls: log_probs[cls])

    def evaluate_standard(self, pos_test, neg_test):
        TP = FP = TN = FN = 0

        for doc in pos_test:
            if self.classify_standard(doc) == 'positive':
                TP += 1
            else:
                FN += 1

        for doc in neg_test:
            if self.classify_standard(doc) == 'negative':
                TN += 1
            else:
                FP += 1

        total = TP + FP + TN + FN
        accuracy  = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
        }

    def evaluate(self, pos_test, neg_test):
        TP = FP = TN = FN = 0

        for doc in pos_test:
            if self.classify(doc) == 'positive':
                TP += 1
            else:
                FN += 1

        for doc in neg_test:
            if self.classify(doc) == 'negative':
                TN += 1
            else:
                FP += 1

        total = TP + FP + TN + FN
        accuracy  = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
        }