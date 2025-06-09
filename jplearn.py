import numpy as np

class Node:
    """A single node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left child node
        self.right = right          # Right child node
        self.value = value          # Class label if this is a leaf node

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None  # Root node of the tree

    def fit(self, X, y):
        """Build the tree from the training data."""
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """Predict class labels for a batch of samples."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or 
            num_classes == 1 or 
            num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Try all features and thresholds to find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            # No good split was found
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        # Recursively build child nodes
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child)

    def _find_best_split(self, X, y):
        """Find the feature and threshold that gives the best Gini gain."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        num_features = X.shape[1]

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue  # skip useless splits

                gain = self._gini_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, y):
        """Calculate the Gini impurity for a set of labels."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _gini_gain(self, parent, left, right):
        """Calculate the information gain from a potential split."""
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        return self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))

    def _most_common_label(self, y):
        """Return the most frequent label in y."""
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict_one(self, x, node):
        """Predict the label for a single sample."""
        while node.value is None:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

