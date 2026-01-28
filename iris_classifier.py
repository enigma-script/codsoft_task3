import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 1. Load Dataset

data = pd.read_csv("IRIS.csv")

print("\nFirst 5 rows:")
print(data.head())


# 2. EDA - Visualizations

sns.pairplot(data, hue="species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

plt.figure()
sns.countplot(x="species", data=data)
plt.title("Class Distribution")
plt.show()


# 3. Feature & Target

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

le = LabelEncoder()
y = le.fit_transform(y)


# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Multiple Models

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

accuracies = {}

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print(f"{name}: {acc:.4f}")


# 6. Accuracy Comparison Plot

plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.ylim(0.9, 1.0)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()


# 7. Best Model Selection

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")


# 8. Confusion Matrix for Best Model

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure()
disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()


# 9. Prediction using Best Model

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

prediction = best_model.predict(new_flower)
print("\nPredicted Species:", le.inverse_transform(prediction)[0])

print("\n✅ Multi-model ML pipeline complete!")
