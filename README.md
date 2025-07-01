# Elevate-Labs-AI-ML-internship-task-6
KNN Classification – Iris Dataset Implemented K-Nearest Neighbors (KNN) classifier on the Iris dataset using Scikit-learn. Included feature normalization, training/testing split, model evaluation with accuracy and confusion matrix, and K-value tuning. Also visualized decision boundaries for better understanding of classification regions.

#description
We used the Iris dataset and implemented K-Nearest Neighbors (KNN) using Scikit-learn. Features were normalized with StandardScaler, the data was split into train-test sets, and the model was evaluated using accuracy, confusion matrix, and classification report. We also tested different K values and visualized decision boundaries for interpretation.

#code
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
error_rate = []
for k in range(1, 21):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    pred_k = knn_k.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10,5))
plt.plot(range(1, 21), error_rate, color='blue', linestyle='dashed', marker='o')
plt.title('Error Rate vs K')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()
from matplotlib.colors import ListedColormap

X_reduced = X_scaled[:, :2]  # using first two features
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train_r, y_train_r)

# Plot
h = .02
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', s=20)
plt.title("KNN Decision Boundaries (Using 2 Features)")
plt.show()








