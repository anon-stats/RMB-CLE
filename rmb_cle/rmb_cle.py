"""
MT-CoLE: Multi-Task Clustering and Local Ensembling

Description: A multi-task learning framework that clusters
tasks based on their similarities and trains specialized models for
each cluster to enhance predictive performance.
License: LGPL-2.1 license

====================================================
Date released : 2025-07-21
Version       : 0.0.2
Update        : 2025-09-06
====================================================
"""

import copy
import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from libs._logging import FileHandler, ConsoleHandler
from sklearn.metrics import mean_squared_error, silhouette_score, accuracy_score


def _split_task(X):
    unique_values = np.unique(X[:, -1])
    mapping = {value: index for index, value in enumerate(unique_values)}
    X[:, -1] = np.vectorize(mapping.get)(X[:, -1])
    X_task = X[:, -1]
    X_data = np.delete(X, -1, axis=1).astype(float)
    return X_data, X_task


class MTCoLE(BaseEstimator):
    def __init__(
        self,
        residual_model_cls,
        task_model_cls,
        residual_model_as_cls,
        n_iter_1st,
        n_iter_3rd,
        max_iter,
        learning_rate,
        regression,
        n_clusters=None,
        random_state=111,
        task_to_cluster_input=None,
    ):

        self.residual_model_cls = residual_model_cls
        self.task_model_cls = task_model_cls
        self.residual_model_as_cls = residual_model_as_cls
        self.n_iter_1st = n_iter_1st
        self.n_iter_3rd = n_iter_3rd
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regression = regression
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.task_to_cluster_input = task_to_cluster_input

        # Placeholders
        self.cluster_models_ = {}
        self.task_to_cluster_ = {}
        self.outlier_tasks_ = set()
        self.residual_matrix_ = []
        self.distance_matrix_ = []
        self.linkage_matrix_ = []

        self.fh_logger = FileHandler()
        self.ch_logger = ConsoleHandler()

        np.random.seed(self.random_state)

    def construct_task_clusters(self, X, y, task_ids):

        unique_tasks = np.unique(task_ids)
        T = len(unique_tasks)

        # Edge cases
        if T == 0:
            # No tasks; return one empty cluster
            self.task_to_cluster_ = {}
            return {1: []}
        if T == 1:
            self.task_to_cluster_ = {unique_tasks[0]: 1}
            return {1: [unique_tasks[0]]}

        task_data = {}
        for task in unique_tasks:
            idx = task_ids == task
            task_data[task] = (X[idx], y[idx])

        # Train a residual model per task
        task_models = {}
        for task in unique_tasks:
            X_task, y_task = task_data[task]
            params = {
                "random_state": getattr(self, "random_state", 0),
                "max_iter": getattr(self, "max_iter", 100),
                "learning_rate": getattr(self, "learning_rate", 0.1),
            }
            model = self.residual_model_cls(**params)
            model.fit(X_task, y_task)
            task_models[task] = copy.deepcopy(model)

        # Cross-task error matrix (lower error -> more similar)
        cross_errors = np.zeros((T, T), dtype=float)
        for i, ti in enumerate(unique_tasks):
            Xi, yi = task_data[ti]
            for j, tj in enumerate(unique_tasks):
                # Score model trained on task tj when predicting task ti
                mdl = task_models[tj]
                yhat = mdl.predict(Xi)
                if getattr(self, "regression", True):
                    err = mean_squared_error(yi, yhat)
                else:
                    if hasattr(mdl, "predict_proba"):
                        proba = mdl.predict_proba(Xi)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            pred = (proba[:, 1] >= 0.5).astype(yi.dtype)
                        else:
                            pred = np.argmax(proba, axis=1).astype(yi.dtype)
                    else:
                        pred = yhat
                    acc = accuracy_score(yi, pred)
                    err = 1.0 - acc  # lower "error" means more similar
                cross_errors[i, j] = err

        # Convert to similarities, then distances
        eps = 1e-8
        sim_matrix = 1.0 / (cross_errors + eps)
        self.residual_matrix_ = sim_matrix  # keep for inspection / debugging
        distance_matrix = cosine_distances(sim_matrix)
        # Numerical safety: enforce symmetry
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        self.distance_matrix_ = distance_matrix

        # Hierarchical clustering
        self.linkage_matrix_ = linkage(
            squareform(self.distance_matrix_), method="average"
        )

        # Helper for auto-k
        def _pick_k_auto(distance_matrix, linkage_matrix, T, k_cap=20):
            if T <= 2:
                return min(T, 2)  # with 2 tasks, k=2 is fine; with 1 handled earlier
            Kmax = min(T, k_cap)
            best_k, best_score = 2, -1.0
            # Try k in [2..Kmax] and keep the silhouette-best
            for k_try in range(2, Kmax + 1):
                labels_try = fcluster(linkage_matrix, k_try, criterion="maxclust")
                try:
                    score = silhouette_score(
                        distance_matrix, labels_try, metric="precomputed"
                    )
                except Exception:
                    # In degenerate cases (e.g., all-zero distances), fallback
                    score = -1.0
                if score > best_score:
                    best_score, best_k = score, k_try
            return best_k

        # Decide k
        if getattr(self, "n_clusters", None) in (None, "auto"):
            k = _pick_k_auto(self.distance_matrix_, self.linkage_matrix_, T, k_cap=20)
        else:
            k = int(self.n_clusters)

        # #tasks ⇒ clusters must be ≤ #tasks"
        k = max(1, min(k, T))

        # Final cut and cluster mapping
        cluster_labels = fcluster(self.linkage_matrix_, k, criterion="maxclust")

        clusters = defaultdict(list)
        self.task_to_cluster_ = {}
        for task, label in zip(unique_tasks, cluster_labels):
            clusters[int(label)].append(task)
            self.task_to_cluster_[task] = int(label)

        return clusters

    def fit(self, X, y):

        X, task_ids = _split_task(X)
        if not self.task_to_cluster_input:
            clusters = self.construct_task_clusters(X, y, task_ids)
        else:
            self.ch_logger.info("[MTCoLE] Using provided task_to_cluster mapping.")
            self.task_to_cluster_ = {
                task: cluster_id
                for task, cluster_id in self.task_to_cluster_input.items()
            }
            clusters = defaultdict(list)
            for task, cluster_id in self.task_to_cluster_input.items():
                clusters[cluster_id].append(task)

        # Train final cluster models (on full task data)
        X = np.column_stack((X, task_ids))
        for cluster_id, tasks in clusters.items():
            cluster_mask = np.isin(task_ids, tasks)
            params_mtgb = {
                "random_state": self.random_state,
                "n_iter_1st": self.n_iter_1st,
                "n_iter_2nd": 0,
                "n_iter_3rd": self.n_iter_3rd,
                "max_depth": 1,
                "subsample": 1.0,
                "learning_rate": self.learning_rate,
            }

            params = {
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "early_stopping": False,
            }
            model = (
                self.task_model_cls(**params_mtgb)
                if not self.residual_model_as_cls
                else self.residual_model_cls(**params)
            )
            model.fit(X[cluster_mask], y[cluster_mask])
            self.cluster_models_[cluster_id] = model
            del model
            for task in tasks:
                self.task_to_cluster_[task] = cluster_id
        return self

    def predict(self, X):
        X_input, task_ids = _split_task(X)
        y_pred = np.empty(X.shape[0])

        X_input = X

        for task in np.unique(task_ids):
            idx = task_ids == task

            cluster_id = self.task_to_cluster_.get(task)
            if cluster_id is None:
                raise ValueError(f"No trained model found for task {task}.")
            model = self.cluster_models_[cluster_id]
            y_pred[idx] = model.predict(X_input[idx])

        return y_pred

    def get_params(self, deep=True):
        return {
            "residual_model_cls": self.residual_model_cls,
            "task_model_cls": self.task_model_cls,
            "residual_model_as_cls": self.residual_model_as_cls,
            "n_iter_1st": self.n_iter_1st,
            "n_iter_3rd": self.n_iter_3rd,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "regression": self.regression,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
