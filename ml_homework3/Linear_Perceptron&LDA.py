import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold

# Generate two-dimensional five-class data
X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=42)

# Using StandardScaler to normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define KFold or StratifiedKFold
# We use StratifiedKFold to maintain the proportion of classes in each fold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store results
accuracy_scores_perceptron = []
precision_scores_perceptron = []
recall_scores_perceptron = []
f1_scores_perceptron = []

accuracy_scores_lda = []
precision_scores_lda = []
recall_scores_lda = []
f1_scores_lda = []

fold_num = 1
# Main loop for KFold
for train_index, test_index in kf.split(X, y):
    print(f"Fold {fold_num}:")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Perceptron Model
    perceptron_model = Perceptron(max_iter=10000, random_state=42)
    perceptron_model.fit(X_train, y_train)
    perceptron_predictions = perceptron_model.predict(X_test)

    # LDA Model
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    lda_predictions = lda_model.predict(X_test)

    # Calculate metrics for Perceptron
    accuracy_perceptron = accuracy_score(y_test, perceptron_predictions)
    precision_perceptron = precision_score(y_test, perceptron_predictions, average='macro')
    recall_perceptron = recall_score(y_test, perceptron_predictions, average='macro')
    f1_perceptron = f1_score(y_test, perceptron_predictions, average='macro')
    accuracy_scores_perceptron.append(accuracy_perceptron)
    precision_scores_perceptron.append(precision_perceptron)
    recall_scores_perceptron.append(recall_perceptron)
    f1_scores_perceptron.append(f1_perceptron)

    print("Perceptron:")
    print(f"Accuracy: {accuracy_perceptron}, Precision: {precision_perceptron}, Recall: {recall_perceptron}, F1-score: {f1_perceptron}")

    # Calculate metrics for LDA
    accuracy_lda = accuracy_score(y_test, lda_predictions)
    precision_lda = precision_score(y_test, lda_predictions, average='macro')
    recall_lda = recall_score(y_test, lda_predictions, average='macro')
    f1_lda = f1_score(y_test, lda_predictions, average='macro')
    accuracy_scores_lda.append(accuracy_lda)
    precision_scores_lda.append(precision_lda)
    recall_scores_lda.append(recall_lda)
    f1_scores_lda.append(f1_lda)

    print("LDA:")
    print(f"Accuracy: {accuracy_lda}, Precision: {precision_lda}, Recall: {recall_lda}, F1-score: {f1_lda}")

    # Display decision regions for Perceptron
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z_perceptron = perceptron_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_perceptron = Z_perceptron.reshape(xx.shape)
    plt.contourf(xx, yy, Z_perceptron, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k')
    plt.title(f"Perceptron Decision Boundaries (Fold {fold_num})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Display decision regions for LDA
    Z_lda = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lda = Z_lda.reshape(xx.shape)
    plt.contourf(xx, yy, Z_lda, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k')
    plt.title(f"LDA Decision Boundaries (Fold {fold_num})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    fold_num += 1

#Calculating the average of criteria
print("Average Perceptron Metrics:")
print(f"Accuracy: {np.mean(accuracy_scores_perceptron)}, Precision: {np.mean(precision_scores_perceptron)}, Recall: {np.mean(recall_scores_perceptron)}, F1-score: {np.mean(f1_scores_perceptron)}")

print("Average LDA Metrics:")
print(f"Accuracy: {np.mean(accuracy_scores_lda)}, Precision: {np.mean(precision_scores_lda)}, Recall: {np.mean(recall_scores_lda)}, F1-score: {np.mean(f1_scores_lda)}")

