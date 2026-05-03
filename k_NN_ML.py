import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gc

# Load MNIST dataset
from sklearn.datasets import fetch_openml
minist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = minist.data, minist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=1000, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set: {X_test.shape}")

plt.close('all')
gc.collect()
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=100)
plt.show()

k_values = [1, 3, 5, 7,9, 15, 25, 51, 101]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k={k:>3d}: Accuracy={acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k")
plt.ylabel("Test Accuracy")
plt.title("K-Nearest Neighbors Performance on the MNIST Accuracy vs k")
plt.grid(True)
plt.savefig("mnist_accuracy_vs_k.png", dpi=100)
plt.show()

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.figure_.suptitle("Confusion Matrix")
plt.savefig('confusion_matrix.png', dpi=100)
print(f"Confusion Matrix:\n{disp.confusion_matrix}")
plt.show()