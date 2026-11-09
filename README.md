# Predictive Maintenance for Engine Health

Final-capstone continuation of the submitted interim report.

## Repository contents
- `data/engine_data.csv` — registered source dataset supplied in the capstone materials
- `src/validate_data.py` — schema, completeness, numeric and target validation
- `src/prepare_data.py` — stratified split and ZIP workflow-artifact creation
- `src/train_model.py` — training-only cross-validation, RandomizedSearchCV, hold-out evaluation and model registration
- `artifacts/` — selected model, metadata, full tuning details, experiment log, baseline CV and hold-out metrics
- `app.py` — Streamlit inference interface
- `requirements.txt` — reproducible runtime dependencies
- `.github/workflows/pipeline.yml` — two-job GitHub Actions ML pipeline
- `Muthu_Aravind_Predictive_Maintenance_Notebook.ipynb` — executed analysis and verification notebook

## Run locally
Use Python 3.11. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/validate_data.py
python src/prepare_data.py
python src/train_model.py
streamlit run app.py
```

## Run the notebook
Open `Muthu_Aravind_Predictive_Maintenance_Notebook.ipynb` from the repository root and execute all cells in order. The notebook imports and executes the source modules in this repository. Export the executed notebook to HTML for code-evidence submission.

## GitHub Actions
Push the project to the `main` branch. Job 1 validates the registered CSV, creates a stratified 80/20 split and uploads the ZIP workflow artifact. Job 2 downloads that artifact, runs the training-only tuning and final hold-out evaluation, and commits the refreshed model and experiment evidence to `main`. The workflow uses `GITHUB_TOKEN` with `contents: write`. Repository branch protection or organization policies may need to allow the workflow bot to push.

## Streamlit Community Cloud
After a successful Actions run registers the model artifacts, create a Streamlit Community Cloud app using this repository, branch `main`, and main file `app.py`. Capture the public app URL and genuine screenshots for the final report.

## Interpretation and safety
The supplied data dictionary does not define which numeric class represents engine failure. The app displays numeric class predictions only. Confirm label semantics, sensor units, calibration and equipment operating limits before operational use. This prototype is not an autonomous safety or maintenance-authorisation system.
