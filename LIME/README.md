# LIME in Python

*LIME (Local Interpretable Model-agnostic Explanations)** is a popular technique for explaining the predictions of any machine learning model in an interpretable and faithful manner. It works by approximating the "black box" model with a simpler, interpretable model (like a linear model) locally around a specific prediction.

## In This Tutorial, You Will Learn How To:

1. Load a dataset and train a classification model.
2. Use LIME to explain individual predictions made by this model.
3. Visualize and interpret these local explanations.

---

## Prerequisites

- Python (3.7 or later recommended)
- Required Python libraries:
  - `lime`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`

Install them using:

```bash
pip install lime scikit-learn pandas numpy matplotlib
```

---

## Step-by-Step Guide

### Step 1: Importing Necessary Libraries

```python
import lime
import lime.lime_tabular
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.metadata

# Print library versions
try:
    lime_version = importlib.metadata.version('lime')
    print(f"LIME version: {lime_version}")
except importlib.metadata.PackageNotFoundError:
    print("LIME library is installed, but version could not be determined.")

print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
```

---

### Step 2: Loading and Preparing Data

```python
print("\n--- Loading and Preparing Data ---")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set samples: {X_train.shape[0]}")
print(f"Test set samples: {X_test.shape[0]}")
print(f"Feature names: {feature_names}")
print(f"Class names: {class_names}")
```

---

### Step 3: Training a Black Box Model

```python
print("--- Training the Model ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.\n")
```

---

### Step 4: Creating the LIME Explainer

```python
print("--- Creating LIME Tabular Explainer ---")
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True
)
print("LIME Explainer created.\n")
```

---

### Step 5: Explaining an Individual Prediction

```python
print("--- Explaining a Single Instance ---")
instance_idx = 0
instance_to_explain = X_test[instance_idx]
true_class_idx = y_test[instance_idx]
true_class_name = class_names[true_class_idx]

model_prediction_proba = model.predict_proba(instance_to_explain.reshape(1, -1))
predicted_class_idx = np.argmax(model_prediction_proba)
predicted_class_name = class_names[predicted_class_idx]

print(f"Instance index to explain: {instance_idx}")
print(f"Instance feature values: {instance_to_explain}")
print(f"Instance true class: {true_class_name}")
print(f"Predicted class: {predicted_class_name}")
print(f"Prediction probabilities: {model_prediction_proba[0]}")

predict_fn = lambda x: model.predict_proba(x)

explanation = explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=predict_fn,
    num_features=len(feature_names),
    top_labels=1
)
print("LIME explanation generated.\n")
```

---

### Step 6: Visualizing and Interpreting the Explanation

```python
explained_label_idx = explanation.top_labels[0] if explanation.top_labels else predicted_class_idx

print(f"--- LIME Explanation for Instance {instance_idx} ---")
print(f"Explanation is for class: {class_names[explained_label_idx]}")
print("Feature contribution weights:")
for feature_description, weight in explanation.as_list(label=explained_label_idx):
    print(f"  "{feature_description}": {weight:.4f}")

try:
    fig = explanation.as_pyplot_figure(label=explained_label_idx)
    fig.suptitle(f"LIME Explanation - Instance {instance_idx}\nPredicted Class: {predicted_class_name} (True Class: {true_class_name})", y=1.02)
    plt.tight_layout()
    plt.show()

    html_explanation_file = f"lime_iris_instance_{instance_idx}_explanation.html"
    explanation.save_to_file(html_explanation_file)
    print(f"Explanation saved to HTML file: {html_explanation_file}")
except Exception as e:
    print(f"Could not generate plot or HTML file: {e}")
```

---

## Interpreting the Output

- **Text Output**: Shows feature weights contributing to the prediction.
- **Matplotlib Plot**: Bar chart showing feature contributions.
- **HTML File**: Interactive version of the explanation for detailed inspection.

---

## Conclusion

You have now learned how to use **LIME** to:

- Set up the LimeTabularExplainer.
- Explain an individual prediction.
- Visualize and interpret local explanations.

LIME is a powerful tool to increase transparency and trust in machine learning models.
