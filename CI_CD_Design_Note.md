# CI/CD Design Note: Defect Model Pipeline

## Overview

This project is designed to make the machine learning process more reliable and automated. Instead of manually training and checking a model every time new data arrives, the pipeline automatically checks the data, trains the model, evaluates its performance, and decides whether the new model should replace the existing one.

The complete workflow is:

**Data → Validation → Training → Evaluation → Promotion → Model Registry**

---

## Pipeline

The pipeline has five main stages:

1. **Data Ingestion** – The pipeline reads the incoming training data.
2. **Data Validation** – It checks whether the data is complete and follows the expected format.
3. **Model Training** – A machine learning pipeline preprocesses the data and trains a Logistic Regression model.
4. **Model Evaluation** – The trained model is tested on a fixed evaluation dataset using metrics such as AUC, accuracy, and F1-score.
5. **Model Promotion** – The new model is compared with the current best model. It is promoted only if its performance is good enough and better than the existing model.

This same process is used in the notebook, automated tests, and CI pipeline so that the model behaves consistently in every environment.

---

## The Two Important Gates

The pipeline uses two checks, or **gates**, to prevent bad models from reaching production.

### 1. Data Validation Gate

Before training starts, the incoming data is checked for problems such as:

* Missing required columns
* Too many missing values
* Numerical values outside the expected range
* Invalid categories such as an unexpected work shift

If the data fails these checks, the pipeline stops immediately and does not train a model.

For example, `train_invalid.csv` contained vibration values outside the allowed range and an invalid `"twilight"` shift category. The validation gate detected these problems and blocked the run.

### 2. Model Promotion Gate

Even if the data is valid, the resulting model may not be better than the current model.

The promotion gate therefore checks two things:

* Does the new model meet the minimum AUC threshold?
* Is it better than the current champion model?

Only when both conditions are satisfied is the new model promoted.

For example, the drifted dataset passed the data validation checks, but the model trained on it performed worse than the existing model. Its AUC was **0.859**, compared with **0.879** for the current champion, so it was rejected.

This means the two gates protect against different problems:

**Validation gate → Is the data reliable?**

**Promotion gate → Is the new model actually better?**

---

## Experiment Tracking and Model Registry

Every pipeline run is recorded in `runs.csv`. The record contains information such as the run ID, number of rows, model metrics, decision, model version, and reason for the decision.

Each trained model is also saved with its own version in the model registry. This makes it possible to identify exactly which model was trained and why it was promoted or rejected.

The current production model is tracked separately as the **champion model**.

---

## CI – Continuous Integration

CI automatically checks the project whenever new code is pushed or a pull request is created.

The CI workflow:

1. Installs the required dependencies.
2. Runs the automated tests using `pytest`.
3. Runs the ML pipeline on the available data.
4. Checks whether everything works correctly.

A rejected model does **not** mean the code is broken. It simply means the model did not perform well enough to replace the current champion.

However, invalid input data causes the build to fail because it indicates a data-quality problem that needs attention.

---

## CD – Continuous Deployment

CD handles the final step of putting a newly promoted model into production.

A new model is deployed only when:

* The CI checks pass.
* The model has actually been promoted.

Therefore, simply having a successful build does not automatically replace the production model.

If a problem is discovered with the latest model, the system can roll back to an earlier model version instead of retraining everything from scratch.

---

## Reproducibility

The pipeline is designed so that the same input should produce the same result.

This is supported by:

* Fixed random seeds
* Pinned package versions in `requirements.txt`
* A fixed evaluation dataset (`eval.csv`)
* Versioned model files
* Recorded experiment results

Using the same evaluation dataset for every model also makes it easier to compare models fairly.

---

## When the Pipeline Runs

The pipeline can run automatically when:

* New code is pushed to the repository
* A pull request is created
* New training data becomes available
* A scheduled retraining process is triggered

If monitoring shows that the incoming data has changed significantly from the data used to train the current model, the model can also be retrained and evaluated again.

---

## Limitations and Future Improvements

This project currently uses synthetic tabular data and mainly focuses on AUC for model comparison.

A more complete production system could be improved by:

* Using multiple evaluation metrics
* Checking model performance for different groups or machine types
* Adding automatic data-drift monitoring
* Monitoring the model after deployment
* Adding human approval for important model updates
* Connecting the pipeline to a real model-serving system

Overall, the project demonstrates how machine learning can be moved from simply **training a model** to building a more reliable and automated **ML production workflow**.

