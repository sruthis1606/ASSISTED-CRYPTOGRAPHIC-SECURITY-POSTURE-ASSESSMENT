from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    canonicalize_dataset,
    generate_sample_dataset,
    is_weak_key,
    train_and_save,
)


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
DEMO_DATASET = DATASETS_DIR / "email_security_demo.csv"
UPLOADED_DATASET = DATASETS_DIR / "uploaded_dataset.csv"
UPLOAD_NAME_FILE = DATASETS_DIR / "uploaded_dataset_name.txt"
MODEL_PATH = MODELS_DIR / "securemailscope.joblib"

DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ASSESSMENT_OPTIONS = {
    "tls_version": ("TLS 1.0", "TLS 1.1", "TLS 1.2", "TLS 1.3"),
    "encryption_mechanism": ("Weak", "AES", "AES-GCM", "ChaCha20"),
    "cryptographic_algorithm": ("DES", "3DES", "RSA", "AES", "ECDSA", "Ed25519"),
    "certificate_information": ("Valid", "Invalid", "Expired", "Self-Signed"),
    "authentication_method": ("Basic", "Password", "OAuth", "Multi-Factor"),
    "cipher_suite": ("Weak", "Moderate", "Strong"),
    "certificate_validity": ("Valid", "Expires within 30 days", "Expired", "Invalid"),
}
CHS_REFERENCE_METRICS = {
    "mae": 1.14,
    "mse": 3.92,
    "rmse": 1.98,
    "r2_score": 0.912,
}
CHS_REFERENCE_POINTS = [
    {"actual": 14, "predicted": 17},
    {"actual": 15, "predicted": 21},
    {"actual": 16, "predicted": 19},
    {"actual": 20, "predicted": 23},
    {"actual": 22, "predicted": 35},
    {"actual": 22, "predicted": 23},
    {"actual": 23, "predicted": 30},
    {"actual": 24, "predicted": 25},
    {"actual": 24, "predicted": 23},
    {"actual": 26, "predicted": 28},
    {"actual": 28, "predicted": 28},
    {"actual": 30, "predicted": 31},
    {"actual": 30, "predicted": 33},
    {"actual": 31, "predicted": 32},
    {"actual": 33, "predicted": 42},
    {"actual": 33, "predicted": 31},
    {"actual": 34, "predicted": 35},
    {"actual": 35, "predicted": 33},
    {"actual": 37, "predicted": 37},
    {"actual": 38, "predicted": 38},
    {"actual": 39, "predicted": 39},
    {"actual": 39, "predicted": 42},
    {"actual": 40, "predicted": 46},
    {"actual": 40, "predicted": 40},
    {"actual": 42, "predicted": 39},
    {"actual": 43, "predicted": 48},
    {"actual": 44, "predicted": 48},
    {"actual": 45, "predicted": 48},
    {"actual": 46, "predicted": 44},
    {"actual": 46, "predicted": 49},
    {"actual": 47, "predicted": 46},
    {"actual": 48, "predicted": 50},
    {"actual": 49, "predicted": 47},
    {"actual": 50, "predicted": 50},
    {"actual": 50, "predicted": 45},
    {"actual": 52, "predicted": 53},
    {"actual": 54, "predicted": 49},
    {"actual": 56, "predicted": 57},
    {"actual": 58, "predicted": 55},
    {"actual": 58, "predicted": 53},
    {"actual": 58, "predicted": 58},
]

if not DEMO_DATASET.exists():
    generate_sample_dataset(DEMO_DATASET)

if not MODEL_PATH.exists():
    try:
        train_and_save(DEMO_DATASET, MODEL_PATH)
    except ValueError as exc:
        print(f"Default demo model could not be trained: {exc}")

CURRENT_DATASET_PATH = UPLOADED_DATASET if UPLOADED_DATASET.exists() else DEMO_DATASET
CURRENT_DATASET_IS_DEMO = CURRENT_DATASET_PATH == DEMO_DATASET
CURRENT_DATASET_NAME = (
    UPLOAD_NAME_FILE.read_text(encoding="utf-8").strip()
    if CURRENT_DATASET_PATH == UPLOADED_DATASET and UPLOAD_NAME_FILE.exists()
    else CURRENT_DATASET_PATH.name
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def read_current_dataset() -> pd.DataFrame:
    try:
        frame = pd.read_csv(CURRENT_DATASET_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the selected CSV dataset: {exc}") from exc
    return canonicalize_dataset(frame)


def load_model_artifact() -> tuple[dict[str, Any] | None, str | None]:
    if not MODEL_PATH.is_file():
        return None, "No trained model is available. Train a model from the Model Training page."
    try:
        artifact = joblib.load(MODEL_PATH)
    except (
        OSError,
        EOFError,
        ValueError,
        KeyError,
        ImportError,
        AttributeError,
        pickle.UnpicklingError,
    ) as exc:
        return None, f"The saved model could not be loaded: {exc}"
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        return None, "The saved model file does not contain a valid SecureMailScope model."
    return artifact, None


def text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def configuration_findings(config: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    tls = text_value(config.get("tls_version")).lower()
    tls_numbers = re.search(r"(\d+(?:\.\d+)?)", tls)
    if not tls:
        findings.append({
            "severity": "high",
            "issue": "TLS version not specified",
            "detail": "The configuration does not identify the negotiated TLS version.",
            "recommendation": "Confirm the negotiated protocol and require TLS 1.2 or newer.",
        })
    elif tls_numbers and float(tls_numbers.group(1)) < 1.2:
        findings.append({
            "severity": "high",
            "issue": "Outdated TLS version",
            "detail": f"{config['tls_version']} is obsolete and does not provide current transport protection.",
            "recommendation": "Disable TLS 1.0/1.1 and require TLS 1.2 or TLS 1.3.",
        })

    algorithm = text_value(config.get("cryptographic_algorithm"))
    if re.search(r"\b(?:3DES|DES|MD5|SHA[\s-]?1|RC4)\b", algorithm, re.IGNORECASE):
        findings.append({
            "severity": "high",
            "issue": "Weak or deprecated cryptographic algorithm",
            "detail": f"{algorithm} is considered weak or deprecated for modern email security.",
            "recommendation": "Replace it with a currently approved algorithm such as AES-GCM or ChaCha20-Poly1305.",
        })

    encryption = text_value(config.get("encryption_mechanism")).lower()
    if any(token in encryption for token in ("weak", "none", "des", "clear")):
        findings.append({
            "severity": "high",
            "issue": "Weak encryption mechanism",
            "detail": f"The configured encryption mechanism is {config['encryption_mechanism']}.",
            "recommendation": "Use authenticated encryption such as AES-GCM or ChaCha20-Poly1305.",
        })

    certificate = text_value(config.get("certificate_information")).lower()
    validity = text_value(config.get("certificate_validity")).lower()
    cert_problem = next(
        (term for term in ("expired", "invalid", "self-signed", "self signed")
         if term in certificate or term in validity),
        None,
    )
    if cert_problem:
        findings.append({
            "severity": "high",
            "issue": "Invalid, expired, or self-signed certificate",
            "detail": "The supplied certificate status indicates it may not establish trusted identity.",
            "recommendation": "Install a valid certificate from a trusted authority and verify its chain and hostname.",
        })
    elif any(term in validity for term in ("soon", "expir", "30 day", "14 day")):
        findings.append({
            "severity": "medium",
            "issue": "Certificate renewal due soon",
            "detail": "The certificate is approaching its expiry date.",
            "recommendation": "Renew the certificate before expiry and confirm automated renewal is monitored.",
        })

    try:
        key_strength = float(config.get("key_strength"))
    except (TypeError, ValueError):
        key_strength = math.nan
    if math.isnan(key_strength):
        findings.append({
            "severity": "medium",
            "issue": "Key strength not specified",
            "detail": "The key length was not provided, so its suitability could not be assessed.",
            "recommendation": "Document key type and size; use at least RSA 2048-bit or an approved elliptic-curve key.",
        })
    elif is_weak_key(algorithm, key_strength):
        if re.search(r"\b(?:ECDSA|ECC|ED25519)\b", algorithm, re.IGNORECASE):
            required_size = "256-bit elliptic-curve"
        elif re.search(r"\b(?:AES|CHACHA20|CHACHA)\b", algorithm, re.IGNORECASE):
            required_size = "128-bit symmetric"
        else:
            required_size = "2048-bit RSA"
        findings.append({
            "severity": "high",
            "issue": "Weak cryptographic key strength",
            "detail": f"The supplied key strength is {key_strength:g} bits.",
            "recommendation": f"Use at least a {required_size} key, or a stronger approved equivalent.",
        })

    authentication = text_value(config.get("authentication_method")).lower()
    if any(term in authentication for term in ("basic", "none", "no auth", "unauthenticated")):
        findings.append({
            "severity": "high",
            "issue": "Inadequate authentication",
            "detail": f"{config['authentication_method']} does not provide strong account protection.",
            "recommendation": "Require modern authenticated submission and enable multi-factor authentication where supported.",
        })
    elif authentication in {"password", "password-only", "password only"}:
        findings.append({
            "severity": "medium",
            "issue": "Password-only authentication",
            "detail": "A password-only account is more exposed to credential theft and reuse.",
            "recommendation": "Add multi-factor authentication and use unique, long credentials.",
        })

    cipher_suite = text_value(config.get("cipher_suite")).lower()
    if any(term in cipher_suite for term in ("weak", "rc4", "null", "export", "40-bit")):
        findings.append({
            "severity": "high",
            "issue": "Weak cipher suite",
            "detail": f"The cipher-suite value ({config['cipher_suite']}) indicates weak or obsolete protection.",
            "recommendation": "Disable weak suites and prefer modern AEAD suites with forward secrecy.",
        })
    elif "moderate" in cipher_suite:
        findings.append({
            "severity": "medium",
            "issue": "Cipher suite needs review",
            "detail": "The cipher suite is marked moderate rather than strong.",
            "recommendation": "Review server policy and prefer modern AEAD suites with forward secrecy.",
        })

    return findings


def security_score(config: dict[str, Any]) -> int:
    penalties = {"high": 19, "medium": 9}
    score = 100 - sum(penalties[item["severity"]] for item in configuration_findings(config))
    return max(0, min(100, score))


def posture_from_score(score: int) -> str:
    if score < 60:
        return "Low"
    if score < 85:
        return "Medium"
    return "High"


def build_dataset_risk_summary(frame: pd.DataFrame) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        config = row.to_dict()
        findings = configuration_findings(config)
        for item in findings:
            issue_counts[item["issue"]] = issue_counts.get(item["issue"], 0) + 1
        if findings:
            records.append({
                "record": int(index) + 1,
                "posture": text_value(row.get(TARGET_COLUMN)),
                "score": security_score(config),
                "issues": ", ".join(item["issue"] for item in findings[:2]),
                "severity": "high" if any(item["severity"] == "high" for item in findings) else "medium",
            })
    return {
        "issue_counts": [
            {"issue": issue, "count": count}
            for issue, count in sorted(issue_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ],
        "records": records[:25],
    }


def build_risk_map_data(frame: pd.DataFrame) -> dict[str, Any]:
    model, model_error = load_model_artifact()
    model_is_current = bool(
        model and model.get("dataset_name") == CURRENT_DATASET_NAME
    )
    selected = frame.head(120).reset_index(drop=True)
    features = selected[FEATURE_COLUMNS]
    predictions: list[str] | None = None
    probabilities: np.ndarray | None = None
    model_name = model.get("model_name") if model_is_current and model else None

    if model_is_current and model:
        pipeline = model["pipeline"]
        predictions = [str(value) for value in pipeline.predict(features)]
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(features)

    risk_for_posture = {"Low": "high", "Medium": "medium", "High": "low"}
    rng = np.random.default_rng(462)
    latitudes = rng.uniform(19.7, 22.1, size=len(selected))
    longitudes = rng.uniform(78.4, 80.7, size=len(selected))
    assets: list[dict[str, Any]] = []

    for position, row in selected.iterrows():
        config = row.to_dict()
        findings = configuration_findings(config)
        score = security_score(config)
        prediction = predictions[position] if predictions is not None else None
        risk_level = (
            risk_for_posture.get(prediction, "medium")
            if prediction is not None
            else ("high" if score < 60 else "medium" if score < 85 else "low")
        )
        confidence = None
        if probabilities is not None and model:
            classifier = model["pipeline"].named_steps["classifier"]
            classes = [str(value) for value in classifier.classes_]
            if prediction in classes:
                confidence = round(float(probabilities[position][classes.index(prediction)]), 4)

        deprecated = []
        for field in ("cryptographic_algorithm", "encryption_mechanism", "cipher_suite"):
            value = text_value(config.get(field))
            if value and re.search(r"\b(?:3DES|DES|MD5|SHA[\s-]?1|RC4)\b", value, re.IGNORECASE):
                deprecated.append(value)

        certificate_information = text_value(config.get("certificate_information"))
        certificate_validity = text_value(config.get("certificate_validity"))
        certificate_status = " · ".join(
            value for value in (certificate_information, certificate_validity) if value
        ) or "Not provided"
        asset_id = 462 + position
        assets.append({
            "asset_id": asset_id,
            "asset_name": (
                "Global Key Management Server"
                if position == 0
                else f"Synthetic Crypto Asset {asset_id}"
            ),
            "latitude": round(float(latitudes[position]), 5),
            "longitude": round(float(longitudes[position]), 5),
            "coordinates_synthetic": True,
            "risk_level": risk_level,
            "posture_prediction": prediction,
            "model_probability": confidence,
            "model_probability_label": (
                f"Probability assigned to predicted {prediction} posture"
                if confidence is not None
                else None
            ),
            "risk_source": (
                f"{model_name} classifier prediction"
                if prediction is not None
                else "Rule-based score fallback"
            ),
            "rule_based_score": score,
            "configuration_checks": (
                "High-severity configuration checks flagged; this is not a compliance audit"
                if any(item["severity"] == "high" for item in findings)
                else "Configuration checks flagged; this is not a compliance audit"
                if findings
                else "No included configuration checks flagged; this is not proof of compliance"
            ),
            "deprecated_algorithms": deprecated,
            "certificate_status": certificate_status,
            "pqc_readiness": "Not assessed — PQC key/protocol fields are not in the dataset",
            "key_usage": "Not reported by the dataset",
            "rng_status": "Not assessed — RNG/FIPS fields are not in the dataset",
            "configuration": {
                column: (
                    float(config[column])
                    if column == "key_strength" and pd.notna(config.get(column))
                    else text_value(config.get(column)) or None
                )
                for column in FEATURE_COLUMNS
            },
        })

    return {
        "assets": assets,
        "asset_count": len(assets),
        "dataset_name": CURRENT_DATASET_NAME,
        "dataset_is_demo": CURRENT_DATASET_IS_DEMO,
        "model_prediction_active": predictions is not None,
        "model_name": model_name,
        "model_message": (
            f"Risk colors use predictions from {model_name}."
            if predictions is not None
            else (
                "The saved model is unavailable or was trained on another dataset; risk colors use the transparent rule-based score."
                if model_error or model
                else "No trained model is available; risk colors use the transparent rule-based score."
            )
        ),
        "source_note": (
            "Asset configurations come from the active dataset. All map coordinates are synthetic display placements, "
            "not measured or actual asset locations. PQC readiness, RNG/FIPS status, and key usage are not assessed "
            "because the current dataset does not contain those fields."
        ),
    }


def dataset_overview(frame: pd.DataFrame) -> dict[str, Any]:
    posture_counts = {
        level: int((frame[TARGET_COLUMN] == level).sum())
        for level in ("Low", "Medium", "High")
    }
    scores = [security_score(row.to_dict()) for _, row in frame.iterrows()]
    count = len(frame)
    model, model_error = load_model_artifact()
    metrics = model.get("metrics", {}) if model else {}
    comparison = model.get("comparison", []) if model else []
    landscape_rows = frame.head(120)
    landscape = [
        {
            "record": int(index) + 1,
            "posture": text_value(row[TARGET_COLUMN]),
            "score": security_score(row.to_dict()),
        }
        for index, row in landscape_rows.iterrows()
    ]
    return {
        "dataset": CURRENT_DATASET_NAME,
        "is_demo": CURRENT_DATASET_IS_DEMO,
        "rows": count,
        "columns": int(frame.shape[1]),
        "mean_score": round(float(np.mean(scores)), 1) if scores else 0,
        "posture_counts": posture_counts,
        "risk_counts": {
            "low": posture_counts["High"],
            "medium": posture_counts["Medium"],
            "high": posture_counts["Low"],
        },
        "model": {
            "available": model is not None,
            "error": model_error,
            "name": model.get("model_name") if model else None,
            "algorithm": model.get("algorithm") if model else None,
            "is_current": model.get("dataset_name") == CURRENT_DATASET_NAME if model else False,
            "metrics": metrics,
            "comparison": comparison,
            "trained_on": model.get("dataset_name") if model else None,
            "training_rows": model.get("training_rows") if model else None,
            "test_samples": model.get("test_samples") if model else None,
        },
        "landscape": landscape,
    }


def describe_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    missing = []
    statistics = []
    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())
        missing.append({
            "feature": column,
            "count": missing_count,
            "percent": round((missing_count / max(len(frame), 1)) * 100, 1),
        })
        entry: dict[str, Any] = {
            "feature": column,
            "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "count": int(series.count()),
        }
        if pd.api.types.is_numeric_dtype(series):
            values = series.describe()
            entry.update({
                key: round(float(values[key]), 2) if pd.notna(values[key]) else None
                for key in ("mean", "std", "min", "25%", "50%", "75%", "max")
            })
        else:
            mode = series.mode(dropna=True)
            entry["unique"] = int(series.nunique(dropna=True))
            entry["top"] = text_value(mode.iloc[0]) if not mode.empty else ""
            entry["top_frequency"] = int((series == mode.iloc[0]).sum()) if not mode.empty else 0
        statistics.append(entry)
    preview_frame = frame.head(10)
    preview = preview_frame.astype(object).where(pd.notna(preview_frame), None).to_dict(orient="records")
    return {
        "dataset": CURRENT_DATASET_NAME,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing": missing,
        "statistics": statistics,
        "preview": preview,
        "features": FEATURE_COLUMNS,
    }


def feature_analysis(frame: pd.DataFrame, selected_feature: str) -> dict[str, Any]:
    if selected_feature not in frame.columns:
        selected_feature = FEATURE_COLUMNS[0]
    series = frame[selected_feature].dropna()
    counts = series.astype(str).value_counts().head(20)
    encoded = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in encoded.columns:
        if not pd.api.types.is_numeric_dtype(encoded[column]):
            categories = sorted(encoded[column].dropna().astype(str).unique())
            mapping = {category: number for number, category in enumerate(categories)}
            encoded[column] = encoded[column].astype(str).map(mapping)
    correlations = encoded.corr(method="pearson").replace({np.nan: 0}).round(3)
    numeric_description: dict[str, Any] = {}
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        numeric_description = {
            key: round(float(desc[key]), 2)
            for key in ("mean", "std", "min", "50%", "max")
        }
    return {
        "feature": selected_feature,
        "labels": [str(value) for value in counts.index],
        "counts": [int(value) for value in counts.values],
        "raw_values": series.tolist()[:500] if pd.api.types.is_numeric_dtype(series) else [],
        "correlation_labels": list(correlations.columns),
        "correlation": correlations.values.tolist(),
        "numeric_description": numeric_description,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/dashboard")
def api_dashboard():
    try:
        return jsonify(dataset_overview(read_current_dataset()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/dataset")
def api_dataset():
    try:
        return jsonify(describe_dataset(read_current_dataset()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/analysis")
def api_analysis():
    try:
        frame = read_current_dataset()
        selected = request.args.get("feature", FEATURE_COLUMNS[0])
        return jsonify(feature_analysis(frame, selected))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/chs-evaluation")
def api_chs_evaluation():
    return jsonify({
        "metrics": CHS_REFERENCE_METRICS,
        "points": CHS_REFERENCE_POINTS,
        "source": "provided_reference_demo",
        "actual_test_evaluation": False,
        "metric_note": (
            "Reference/demo values from the supplied CHS design. The current model predicts "
            "Low/Medium/High categories and has no numeric CHS test target."
        ),
        "plot_note": (
            "Illustrative reference/demo points only; they are not predictions or test records "
            "from the current classifier."
        ),
    })


@app.get("/api/risks")
def api_risks():
    try:
        return jsonify(build_dataset_risk_summary(read_current_dataset()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/risk-map")
def api_risk_map():
    try:
        return jsonify(build_risk_map_data(read_current_dataset()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/upload")
def api_upload():
    global CURRENT_DATASET_PATH, CURRENT_DATASET_IS_DEMO, CURRENT_DATASET_NAME
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a CSV file before uploading."}), 400
    safe_name = secure_filename(uploaded.filename)
    if not safe_name.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv dataset files are supported."}), 400
    try:
        frame = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return jsonify({"error": f"Could not parse this CSV file: {exc}"}), 400
    try:
        frame = canonicalize_dataset(frame)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    UPLOADED_DATASET.parent.mkdir(exist_ok=True)
    frame.to_csv(UPLOADED_DATASET, index=False)
    UPLOAD_NAME_FILE.write_text(safe_name, encoding="utf-8")
    CURRENT_DATASET_PATH = UPLOADED_DATASET
    CURRENT_DATASET_IS_DEMO = False
    CURRENT_DATASET_NAME = safe_name
    return jsonify({
        "message": "Dataset uploaded and validated. Train a model to evaluate it.",
        "filename": safe_name,
        "rows": len(frame),
        "columns": len(frame.columns),
    })


@app.get("/download/dataset")
def download_dataset():
    try:
        read_current_dataset()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return send_file(
        CURRENT_DATASET_PATH,
        as_attachment=True,
        download_name=CURRENT_DATASET_NAME,
        mimetype="text/csv",
    )


@app.get("/download/demo")
def download_demo():
    return send_file(
        DEMO_DATASET,
        as_attachment=True,
        download_name=DEMO_DATASET.name,
        mimetype="text/csv",
    )


@app.post("/api/train")
def api_train():
    try:
        frame = read_current_dataset()
        result = train_and_save(CURRENT_DATASET_PATH, MODEL_PATH, dataset_name=CURRENT_DATASET_NAME)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "message": "Model comparison completed and the best-performing model was saved.",
        "selected_model": result["model_name"],
        "metrics": result["metrics"],
        "comparison": result["comparison"],
        "rows": len(frame),
    })


@app.post("/api/assess")
def api_assess():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Submit the security configuration as a JSON object."}), 400
    config = {key: payload.get(key) for key in FEATURE_COLUMNS}
    missing = [key.replace("_", " ") for key, value in config.items() if value is None or not str(value).strip()]
    if missing:
        return jsonify({"error": f"Complete all security fields: {', '.join(missing)}."}), 400
    for field, allowed in ASSESSMENT_OPTIONS.items():
        submitted = str(config[field]).strip()
        match = next((option for option in allowed if option.casefold() == submitted.casefold()), None)
        if match is None:
            return jsonify({"error": f"Choose a supported value for {field.replace('_', ' ')}."}), 400
        config[field] = match
    try:
        key_strength = float(config["key_strength"])
    except (TypeError, ValueError):
        return jsonify({"error": "Key strength must be a number of bits."}), 400
    if not math.isfinite(key_strength) or key_strength < 1 or key_strength > 16384:
        return jsonify({"error": "Key strength must be between 1 and 16,384 bits."}), 400
    if not key_strength.is_integer():
        return jsonify({"error": "Key strength must be a whole number of bits."}), 400
    config["key_strength"] = key_strength

    artifact, model_error = load_model_artifact()
    if artifact is None:
        return jsonify({"error": model_error}), 503
    if artifact.get("dataset_name") != CURRENT_DATASET_NAME:
        return jsonify({
            "error": "The saved model was trained on a different dataset. Retrain the models before running this assessment."
        }), 409
    input_frame = pd.DataFrame([config], columns=FEATURE_COLUMNS)
    pipeline = artifact["pipeline"]
    prediction = str(pipeline.predict(input_frame)[0])
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_frame)[0]
        classes = pipeline.named_steps["classifier"].classes_
        confidence = round(float(max(probabilities)), 4)
        class_probabilities = {
            str(name): round(float(probability), 4)
            for name, probability in zip(classes, probabilities)
        }
    else:
        class_probabilities = {}
    score = security_score(config)
    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "score": score,
        "score_posture": posture_from_score(score),
        "findings": configuration_findings(config),
        "model_name": artifact.get("model_name", "Trained classifier"),
        "model_note": "Posture is predicted by the trained classifier; the 0-100 score and findings use transparent configuration checks.",
    })


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The uploaded CSV is too large. The maximum supported file size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
