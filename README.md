# Production ML Pipeline with CI/CD

An end-to-end ML pipeline that automates **data validation, model training, evaluation, versioning, and model promotion** with CI/CD.

###  Workflow

**Data → Validation → Training → Evaluation → Promotion → Model Registry**

###  Key Features

* **Data Validation Gate** – blocks invalid or malformed training data before training.
* **Model Evaluation** – evaluates models using AUC, accuracy, and F1-score on a fixed evaluation set.
* **Promotion Gate** – promotes a new model only if it meets the performance threshold and outperforms the current champion.
* **Experiment Tracking** – records pipeline runs, metrics, decisions, and model versions.
* **CI/CD** – automated testing and pipeline execution using GitHub Actions.

### Tested Scenarios

The pipeline was tested with **baseline, invalid, drifted, and improved datasets** to verify data validation and model promotion behaviour.

###  Tech Stack

**Python · Scikit-learn · Pandas · Pytest · Jupyter · GitHub Actions**

###  Main Files

* `production-ml-pipeline-ci-cd.ipynb` – complete workflow
* `mlpipe.py` – pipeline logic
* `test_pipeline.py` – automated tests
* `run_pipeline.py` – pipeline execution
* `CI_CD_Design_Note.md` – CI/CD architecture and design
* `data/` – training and evaluation datasets
* `registry/` – versioned models and champion tracking
