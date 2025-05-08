# SIOP_XAI

This document serves as a **supplement to the SIOP Machine Learning Handbook for Explainable AI (XAI)**. It introduces two widely used model explanation techniques: **LIME** and **SHAP**.

## Overview

- **LIME (Local Interpretable Model-agnostic Explanations)**  
  A model-agnostic local explanation method that fits an interpretable (usually linear) model around a single prediction by sampling nearby points.

- **SHAP (SHapley Additive exPlanations)**  
  A game-theory-based method using Shapley values to assign feature importance globally and locally across a model's predictions.

## More Information

To explore the latest documentation, usage, and updates, refer to their official GitHub repositories:

- [LIME GitHub Repository](https://github.com/marcotcr/lime)
- [SHAP GitHub Repository](https://github.com/shap/shap)

## What This Supplement Includes

For each method, we provide a practical example, including:

- A brief description of the dataset used (e.g., Iris, California Housing)
- The machine learning model (e.g., RandomForestClassifier, RandomForestRegressor)
- A complete step-by-step guide on applying the XAI method
- Explanation and visualization of the outputs
- Fully functional Python code that can be run directly

## Included Files

- `SHAP`  
  A beginner-friendly guide for using SHAP to explain a regression model.

- `LIME`  
  A beginner-friendly guide for using LIME to explain a classification model.

## Recommendations

We recommend reading this supplement alongside the main handbook. It bridges the theory with hands-on practice and provides a clear, accessible pathway to applying explainability techniques to your own models.

This material is especially useful for teaching, research, model development, and debugging tasks involving black-box ML models.
