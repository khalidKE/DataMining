# Operational notes

## Refresh dataset
- Replace `creditcard.csv` with the updated dataset (keep same schema).  
- If the file exceeds GitHub limits, keep it out of version control and share via secure storage or Git LFS.

## Re-run the pipeline
- `python src/project_pipeline.py --max-samples 100000` (adjust `--max-samples` or set to `0` for full data).  
- The command rewrites `reports/eda_summary.md`, refreshes `plots/*.png`, updates `outputs/metrics.json`, and dumps the newest `{model}.joblib` plus `scaler.joblib`.

## Dashboard
- Ensure dependencies are installed (`pip install -r requirements.txt`).  
- Start Streamlit with `streamlit run app.py`; it auto-pulls the latest metrics/plots for inspection.

## Rebuilding explainability / models
- The training loop already selects the best AUPRC model and saves a SHAP summary (`plots/shap_summary_<model>.png`).  
- To experiment with alternative learners, edit `build_models` inside `src/project_pipeline.py`, rerun the pipeline, and the dashboard will reflect the new artifacts.
