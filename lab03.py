import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from scipy.stats import norm
from scipy.spatial import distance

# ==== Step 1: Create a Simple Elephant Call Dataset ====
np.random.seed(42)
samples_per_class = 40

rumble_calls = np.random.normal(2, 0.5, (samples_per_class, 5))
trumpet_calls = np.random.normal(5, 0.5, (samples_per_class, 5))
roar_calls = np.random.normal(8, 0.5, (samples_per_class, 5))

feature_matrix = np.vstack((rumble_calls, trumpet_calls, roar_calls))
call_labels = ['rumble'] * samples_per_class + ['trumpet'] * samples_per_class + ['roar'] * samples_per_class

data = pd.DataFrame(feature_matrix, columns=['mfcc1', 'mfcc2', 'zcr', 'duration', 'energy'])
data['call_type'] = call_labels
data.to_csv("elephant_calls_dataset.csv", index=False)

# ==== Step 2: Encode Labels and Split ====
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['call_type'])
features = data.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=1)

# ==== Step 3: kNN Model (k=3) ====
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# ==== A1: Class Means, Spreads and Distances ====
group_means = data.groupby('call_type').mean()
group_stds = data.groupby('call_type').std()

distance_rumble_trumpet = np.linalg.norm(group_means.loc['rumble'] - group_means.loc['trumpet'])
distance_rumble_roar = np.linalg.norm(group_means.loc['rumble'] - group_means.loc['roar'])
distance_trumpet_roar = np.linalg.norm(group_means.loc['trumpet'] - group_means.loc['roar'])

print("Class Means:\n", group_means)
print("Class Spreads:\n", group_stds)
print("Class Distances:")
print("Rumble–Trumpet:", distance_rumble_trumpet)
print("Rumble–Roar:", distance_rumble_roar)
print("Trumpet–Roar:", distance_trumpet_roar)

# ==== A2: Histogram of Feature (mfcc1) ====
mfcc1_values = data['mfcc1']
plt.hist(mfcc1_values, bins=10, alpha=0.7, color='skyblue', density=True)
x = np.linspace(mfcc1_values.min(), mfcc1_values.max(), 100)
plt.plot(x, norm.pdf(x, mfcc1_values.mean(), mfcc1_values.std()), 'r-', label='Normal Distribution')
plt.title("Histogram of MFCC1 with Normal Curve")
plt.legend()
plt.show()

print("MFCC1 Mean:", mfcc1_values.mean(), "Variance:", mfcc1_values.var())

# ==== A3: Minkowski Distance Plot ====
sample_1 = features[0]
sample_2 = features[1]
minkowski_distances = [distance.minkowski(sample_1, sample_2, r) for r in range(1, 11)]
plt.plot(range(1, 11), minkowski_distances, marker='o')
plt.title("Minkowski Distance (r = 1 to 10)")
plt.xlabel("r value")
plt.ylabel("Distance")
plt.grid()
plt.show()

# ==== A6: Test Accuracy ====
test_accuracy = knn_classifier.score(X_test, y_test)
print("Test Accuracy (k=3):", test_accuracy)

# ==== A7: Predictions ====
test_predictions = knn_classifier.predict(X_test)
print("Test Predictions:", test_predictions)
print("Prediction for first test sample:", knn_classifier.predict(X_test[0].reshape(1, -1)))

# ==== A8: Accuracy vs k ====
k_values = range(1, 12)
accuracy_scores = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy_scores.append(model.score(X_test, y_test))

plt.plot(k_values, accuracy_scores, marker='s', color='orange')
plt.title("Accuracy vs k in kNN")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# ==== A9: Confusion Matrix & Classification Report ====
print("Confusion Matrix:\n", confusion_matrix(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions, target_names=label_encoder.classes_))

# ==== O2: Accuracy Using Different Distance Metrics ====
for metric_type in ['euclidean', 'manhattan', 'chebyshev']:
    model = KNeighborsClassifier(n_neighbors=3, metric=metric_type)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{metric_type.capitalize()} Distance Accuracy:", score)

# ==== O3: AUROC Curve (only for 2-class problems) ====
if len(label_encoder.classes_) == 2:
    y_test_binary = label_binarize(y_test, classes=[0, 1])
    prediction_probs = knn_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_binary, prediction_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("AUROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
