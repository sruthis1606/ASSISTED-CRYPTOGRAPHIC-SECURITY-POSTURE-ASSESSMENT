# SecureMailScope

SecureMailScope is a local Flask prototype for AI-assisted assessment of email-system cryptographic configurations. It analyzes a labeled CSV, compares classification models, predicts a posture for entered settings, and reports transparent configuration-based checks.

**Scope and data notice:** the first-run dataset is synthetically generated and labeled as demonstration data in the interface. It is not evidence about real servers. Assessments use only values provided by the user; the application does not resolve hostnames, contact email servers, check certificate chains online, or perform live scans.

## Project structure

```text
securemailscope/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── datasets/
│   ├── .gitkeep
│   └── email_security_demo.csv     # generated on first run
├── models/
│   ├── .gitkeep
│   └── securemailscope.joblib      # generated on first run
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

The application creates the demo CSV and the initial model artifact automatically when they are missing.

## Run in VS Code on Windows

Open this folder in VS Code, then open **Terminal → New Terminal**. Run each command separately from the project folder:

```powershell
py -3 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in that terminal, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

To explicitly generate the demonstration dataset and train/compare all models:

```powershell
python train_model.py
```

Start the website in a separate command in the same activated terminal:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. Stop the development server with `Ctrl+C`.

For a CSV that already contains labeled security data:

```powershell
python train_model.py --dataset .\datasets\your_dataset.csv
```

The upload page accepts CSVs containing all eight configuration fields and the target `security_posture`. Column names are normalized, including common forms such as `TLS Version`, `Key Strength`, and `Security Posture`. Rows with missing feature values can be handled by the preprocessing pipeline; target labels must be `Low`, `Medium`, or `High` (common equivalents such as `Moderate` are also recognized).

## Features

- Dashboard statistics and Plotly charts based on the current dataset.
- CSV upload validation, row/column counts, descriptive statistics, missing-value analysis, preview, and download.
- Feature distributions and an explicitly exploratory encoded-correlation heatmap.
- Random Forest, Extra Trees, Gradient Boosting, AdaBoost, and MLP comparison using accuracy, weighted precision, recall, and F1.
- Local Joblib model persistence and one-click retraining.
- Configuration assessment with model probabilities, a separate transparent rule-based score, and targeted recommendations.
- Risk summaries based on actual dataset rows.
- An interactive OpenStreetMap-backed risk map whose asset configurations come from the active dataset and whose map coordinates are clearly labeled synthetic display placements, not real infrastructure locations.
- Map risk colors use the current trained classifier when it matches the active dataset; rule-check fallbacks are identified, and unavailable PQC, RNG/FIPS, key-usage, or certificate-expiry data is shown as not assessed.

Interactive charts load Plotly.js from its CDN; the map loads Leaflet and OpenStreetMap tiles. These visual features need an internet connection. Flask and all data/model processing run locally.
