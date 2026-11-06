# Deploying this app on Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in with the GitHub account that owns this repo.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill the form:
   - Repository: `UDDEISHYA/credit-risk-app`
   - Branch: `main`
   - Main file path: `Home.py`
4. Open **Advanced settings** and set **Python version = 3.12**. This is required: the pinned `scikit-learn==1.4.2` and `numpy==1.26.4` only publish wheels for Python 3.9–3.12. Letting Streamlit default to 3.13 or 3.14 will fail at install time.
5. Click **Deploy**. First build takes ~2–3 minutes.

## Local run

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Pinned dependencies

`requirements.txt` pins `scikit-learn`, `numpy` and `joblib` to the exact versions used to train the model in `artifacts/model_bundle.joblib`. Do not change these pins without re-training and re-serialising the bundle.
