import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tarmaccore.explainers.base import IExplainer


class DeltaXplainer(IExplainer):
    def __init__(self, min_leaf=0.01, seed=0):
        self.min_leaf = min_leaf
        self.seed = seed
        self.feature_names = None

    def fit(self, X, y):
        leaf = max(1, int(self.min_leaf * len(X)))
        self.tree = DecisionTreeClassifier(
            min_samples_leaf=leaf, random_state=self.seed
        )

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:

            if unique_classes[0] == 0:
                dummy_class = 1
            else:
                dummy_class = 0

            if isinstance(X, pd.DataFrame):
                dummy_X = X.iloc[[0]].copy()  # Copy first row
                X = pd.concat([X, dummy_X])
            else:
                dummy_X = X[[0]].copy()  # Copy first row
                X = np.vstack([X, dummy_X])
            y = np.append(y, dummy_class)

        self.tree.fit(X, y)

        if hasattr(X, "columns"):
            self.feature_names = X.columns
        return self

    def format_rule_dict(self, rule):
        """Format a rule into a structured dictionary."""
        conditions = []
        for feat, op, thresh in rule["path"]:
            feat_name = (
                f"feature_{feat}"
                if self.feature_names is None
                else self.feature_names[feat]
            )
            conditions.append(
                {
                    "feature": feat_name,
                    "operator": op,
                    "threshold": round(float(thresh), 3),
                }
            )

        samples = int(rule["samples"])  # Convert np.int64 to Python int
        total_samples = int(self.tree.tree_.n_node_samples[0])
        disagreement_pct = round(float(rule["disagreement_pct"]) * 100, 1)

        return {
            "conditions": conditions,
            "samples_affected": samples,
            "disagreement_percentage": disagreement_pct,
            "prediction": "models differ",
            "support": round(float(samples) / total_samples, 3),
        }

    def format_rule_str(self, rule):
        """Format a rule into a readable string."""
        conditions = []
        for feat, op, thresh in rule["path"]:
            feat_name = (
                f"feature_{feat}"
                if self.feature_names is None
                else self.feature_names[feat]
            )
            conditions.append(f"{feat_name} {op} {thresh:.3f}")

        disagreement_pct = round(float(rule["disagreement_pct"]) * 100, 1)
        return f"IF {' AND '.join(conditions)} THEN models differ (affects {rule['samples']} samples, {disagreement_pct}% disagree)"

    def explain(self, return_dict=False):
        """Explain model differences.

        Args:
            return_dict: If True, return structured dictionaries instead of strings
        """
        tree = self.tree.tree_
        feature = tree.feature
        threshold = tree.threshold
        rules = []

        def recurse(node, path):
            if tree.children_left[node] == -1:  # leaf
                node_samples = tree.n_node_samples[node]
                node_values = tree.value[node][0]

                disagreement_pct = node_values[1] / node_values.sum()

                if (
                    disagreement_pct > 0
                ):  # Include all rules where there's any disagreement

                    simplified_path = []
                    for feat, op, thresh in path:

                        feat_conditions = [
                            (o, t) for f, o, t in simplified_path if f == feat
                        ]

                        if op == ">":

                            if not feat_conditions:
                                simplified_path.append((feat, op, thresh))

                            elif all(
                                o != ">" or t < thresh for o, t in feat_conditions
                            ):

                                simplified_path = [
                                    (f, o, t)
                                    for f, o, t in simplified_path
                                    if f != feat or o != ">"
                                ]
                                simplified_path.append((feat, op, thresh))

                        elif op == "<=":

                            if not feat_conditions:
                                simplified_path.append((feat, op, thresh))

                            elif all(
                                o != "<=" or t > thresh for o, t in feat_conditions
                            ):

                                simplified_path = [
                                    (f, o, t)
                                    for f, o, t in simplified_path
                                    if f != feat or o != "<="
                                ]
                                simplified_path.append((feat, op, thresh))
                    rule = {
                        "path": simplified_path,
                        "samples": node_samples,
                        "disagreement_pct": disagreement_pct,
                    }
                    rules.append(rule)
            else:
                thresh = threshold[node]
                feat = feature[node]
                recurse(tree.children_left[node], path + [(feat, "<=", thresh)])
                recurse(tree.children_right[node], path + [(feat, ">", thresh)])

        recurse(0, [])

        rules.sort(key=lambda x: x["disagreement_pct"] * x["samples"], reverse=True)

        if return_dict:
            return [self.format_rule_dict(rule) for rule in rules]
        else:
            return [self.format_rule_str(rule) for rule in rules]
