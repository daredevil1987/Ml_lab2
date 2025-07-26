
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.stats import norm
from scipy.spatial import distance

# Simulated Elephant Call Data (2 classes: Trumpet=0, Rumble=1)
np.random.seed(0)
trumpet = np.random.normal(5, 1, (50, 5))  # Class 0
rumble = np.random.normal(2, 1, (50, 5))   # Class 1
X = np.vstack((trumpet, rumble))
y = np.array([0]*50 + [1]*50)

# A1: Class Mean, Std, and Distance
mean0 = X[y==0].mean(axis=0)
mean1 = X[y==1].mean(axis=0)
std0 = X[y==0].std(axis=0)
std1 = X[y==1].std(axis=0)
dist = np.linalg.norm(mean0 - mean1)
print("Mean Class 0:", mean0)
print("Mean Class 1:", mean1)
print("Spread Class 0:", std0)
print("Spread Class 1:", std1)
print("Distance between classes:", dist)

# A2: Histogram of First Feature
feature = X[:, 0]
plt.hist(feature, bins=10, color='skyblue')
plt.title("Histogram of Feature 1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
print("Feature 1 Mean:", feature.mean())
print("Feature 1 Variance:", feature.var())

# A3: Minkowski Distance
x1, x2 = X[0], X[1]
minkowski_dists = [distance.minkowski(x1, x2, r) for r in range(1, 11)]
plt.plot(range(1, 11), minkowski_dists, marker='o')
plt.title("Minkowski Distance (r=1 to 10)")
plt.xlabel("r")
plt.ylabel("Distance")
plt.grid()
plt.show()

# A4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# A5: Train kNN Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# A6: Accuracy
print("Test Accuracy (k=3):", model.score(X_test, y_test))

# A7: Prediction
print("Predictions:", model.predict(X_test))
print("Prediction for first test vector:", model.predict(X_test[0].reshape(1, -1)))

# A8: Accuracy vs K
accs = []
for k in range(1, 12):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    accs.append(m.score(X_test, y_test))
plt.plot(range(1, 12), accs, marker='s')
plt.title("Accuracy vs K")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# A9: Confusion Matrix and Report
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# O1: Histogram vs Normal Distribution
plt.hist(feature, bins=10, density=True, alpha=0.6, label='Histogram')
x_vals = np.linspace(min(feature), max(feature), 100)
plt.plot(x_vals, norm.pdf(x_vals, feature.mean(), np.sqrt(feature.var())), 'r-', label='Normal Curve')
plt.legend()
plt.title("Histogram vs Normal Distribution")
plt.show()

# O2: Accuracy for Different Distance Metrics
for metric in ['euclidean', 'manhattan', 'chebyshev']:
    model_metric = KNeighborsClassifier(n_neighbors=3, metric=metric)
    model_metric.fit(X_train, y_train)
    acc = model_metric.score(X_test, y_test)
    print(f"Accuracy with {metric} distance:", acc)

# O3: AUROC Curve
y_bin = label_binarize(y_test, classes=[0, 1])
probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_bin, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='kNN (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve")
plt.legend()
plt.grid()
plt.show()
