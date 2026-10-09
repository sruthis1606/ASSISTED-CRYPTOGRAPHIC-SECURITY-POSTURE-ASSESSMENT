from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
FEATURE_COLUMNS = [
    "tls_version",
    "encryption_mechanism",
    "cryptographic_algorithm",
    "certificate_information",
    "authentication_method",
    "key_strength",
    "cipher_suite",
    "certificate_validity",
]
TARGET_COLUMN = "security_posture"
MODEL_NAMES = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "MLP Neural Network",
]
ALIASES = {
    "tlsversion": "tls_version",
    "tlsprotocol": "tls_version",
    "encryptionmechanism": "encryption_mechanism",
    "encryptiontype": "encryption_mechanism",
    "cryptographicalgorithm": "cryptographic_algorithm",
    "algorithm": "cryptographic_algorithm",
    "certificateinformation": "certificate_information",
    "certificatestatus": "certificate_information",
    "authenticationmethod": "authentication_method",
    "authenticationtype": "authentication_method",
    "keystrength": "key_strength",
    "keysize": "key_strength",
    "ciphersuite": "cipher_suite",
    "ciphersuites": "cipher_suite",
    "certificatevalidity": "certificate_validity",
    "certificatevaliditystatus": "certificate_validity",
    "securityposture": "security_posture",
    "securitylevel": "security_posture",
    "posture": "security_posture",
}
POSTURE_ALIASES = {
    "low": "Low",
    "poor": "Low",
    "weak": "Low",
    "insecure": "Low",
    "highrisk": "Low",
    "critical": "Low",
    "medium": "Medium",
    "moderate": "Medium",
    "med": "Medium",
    "mediumrisk": "Medium",
    "high": "High",
    "good": "High",
    "strong": "High",
    "safe": "High",
    "secure": "High",
    "lowrisk": "High",
}


def canonicalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    rename: dict[Any, str] = {}
    for column in frame.columns:
        compact = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
        canonical = ALIASES.get(compact)
        if canonical:
            rename[column] = canonical
    normalized = frame.rename(columns=rename)
    duplicated = normalized.columns[normalized.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"More than one uploaded column maps to: {', '.join(duplicated)}.")
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        readable = ", ".join(column.replace("_", " ") for column in missing)
        raise ValueError(
            f"Missing required column(s): {readable}. Include all eight security features and Security Posture."
        )
    normalized = normalized[required].copy()
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].map(
        lambda value: _normalize_posture(value)
    )
    if normalized[TARGET_COLUMN].isna().any():
        missing_rows = int(normalized[TARGET_COLUMN].isna().sum())
        raise ValueError(
            f"Security Posture contains {missing_rows} blank or unsupported value(s). Use Low, Medium, or High."
        )
    normalized["key_strength"] = pd.to_numeric(normalized["key_strength"], errors="coerce")
    if normalized["key_strength"].isna().all():
        raise ValueError("Key Strength must contain at least one numeric value measured in bits.")
    for column in FEATURE_COLUMNS:
        if column != "key_strength":
            normalized[column] = normalized[column].map(
                lambda value: np.nan if pd.isna(value) else str(value).strip()
            )
    return normalized


def _normalize_posture(value: Any) -> str | None:
    if pd.isna(value):
        return None
    compact = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    return POSTURE_ALIASES.get(compact)


def generate_sample_dataset(
    filepath: str | Path | None = None,
    n_samples: int = 480,
    random_seed: int = 42,
) -> pd.DataFrame:
    if n_samples < 30:
        raise ValueError("Generate at least 30 demo records so every posture can be represented.")
    target = Path(filepath) if filepath else DATASETS_DIR / "email_security_demo.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    frame = pd.DataFrame({
        "tls_version": rng.choice(["TLS 1.0", "TLS 1.1", "TLS 1.2", "TLS 1.3"], n_samples, p=[.12, .15, .38, .35]),
        "encryption_mechanism": rng.choice(["Weak", "AES", "AES-GCM", "ChaCha20"], n_samples, p=[.12, .27, .43, .18]),
        "cryptographic_algorithm": rng.choice(["DES", "3DES", "RSA", "AES", "ECDSA"], n_samples, p=[.08, .09, .32, .29, .22]),
        "certificate_information": rng.choice(["Valid", "Invalid", "Expired", "Self-Signed"], n_samples, p=[.72, .08, .11, .09]),
        "authentication_method": rng.choice(["Basic", "Password", "OAuth", "Multi-Factor"], n_samples, p=[.12, .24, .35, .29]),
        "key_strength": rng.choice([512, 1024, 2048, 3072, 4096, 256], n_samples, p=[.04, .10, .36, .23, .17, .10]),
        "cipher_suite": rng.choice(["Weak", "Moderate", "Strong"], n_samples, p=[.16, .34, .50]),
        "certificate_validity": rng.choice(["Valid", "Invalid", "Expired", "Expires within 30 days"], n_samples, p=[.69, .09, .12, .10]),
    })
    scores = frame.apply(_demo_score, axis=1).to_numpy()
    frame[TARGET_COLUMN] = np.where(scores < 60, "Low", np.where(scores < 85, "Medium", "High"))
    if frame[TARGET_COLUMN].nunique() < 3:
        order = np.argsort(scores)
        frame.loc[order[: max(1, n_samples // 4)], TARGET_COLUMN] = "Low"
        frame.loc[order[-max(1, n_samples // 4):], TARGET_COLUMN] = "High"
    frame.to_csv(target, index=False)
    return frame


def _demo_score(row: pd.Series) -> int:
    config = row.to_dict()
    score = 100
    tls = str(config["tls_version"])
    match = re.search(r"(\d+(?:\.\d+)?)", tls)
    if not match or float(match.group(1)) < 1.2:
        score -= 19
    elif float(match.group(1)) < 1.3:
        score -= 3
    if re.search(r"\b(?:3DES|DES|MD5|SHA[\s-]?1)\b", str(config["cryptographic_algorithm"]), re.I):
        score -= 19
    if "weak" in str(config["encryption_mechanism"]).lower():
        score -= 19
    cert = f"{config['certificate_information']} {config['certificate_validity']}".lower()
    if any(term in cert for term in ("invalid", "expired", "self-signed")):
        score -= 19
    elif "30 days" in cert:
        score -= 9
    key = float(config["key_strength"])
    if is_weak_key(config["cryptographic_algorithm"], key):
        score -= 19
    auth = str(config["authentication_method"]).lower()
    if auth == "basic":
        score -= 19
    elif auth == "password":
        score -= 9
    cipher = str(config["cipher_suite"]).lower()
    if cipher == "weak":
        score -= 19
    elif cipher == "moderate":
        score -= 9
    return max(0, score)


def is_weak_key(algorithm: Any, strength: float) -> bool:
    algorithm_name = str(algorithm).upper()
    if re.search(r"\b(?:ECDSA|ECC|ED25519)\b", algorithm_name):
        return strength < 256
    if re.search(r"\b(?:AES|CHACHA20|CHACHA)\b", algorithm_name):
        return strength < 128
    return strength < 2048


def make_preprocessor() -> ColumnTransformer:
    categorical = [column for column in FEATURE_COLUMNS if column != "key_strength"]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
            (
                "numerical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["key_strength"],
            ),
        ],
        remainder="drop",
    )


def model_candidates(random_seed: int = 42) -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=1, class_weight="balanced", random_state=random_seed, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=180, min_samples_leaf=1, class_weight="balanced", random_state=random_seed, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_seed),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=450, early_stopping=False, random_state=random_seed
        ),
    }


def train_and_save(
    dataset_path: str | Path,
    model_path: str | Path,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    source = Path(dataset_path)
    try:
        raw = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read training dataset: {exc}") from exc
    frame = canonicalize_dataset(raw)
    if len(frame) < 15:
        raise ValueError("At least 15 labeled records are required to train and evaluate a classifier.")
    class_counts = frame[TARGET_COLUMN].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two security-posture classes (Low, Medium, or High).")
    if (class_counts < 2).any():
        rare = ", ".join(class_counts[class_counts < 2].index.astype(str))
        raise ValueError(f"At least two training records are needed for each present posture; add more {rare} examples.")
    test_count = max(int(np.ceil(len(frame) * 0.2)), int(len(class_counts)))
    if len(frame) - test_count < len(class_counts):
        raise ValueError("There are too few records to keep every posture in both training and test splits.")

    features = frame[FEATURE_COLUMNS]
    labels = frame[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_count,
        random_state=42,
        stratify=labels,
    )
    results: list[dict[str, Any]] = []
    candidates = model_candidates()
    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        }
        results.append({"name": name, **metrics})
    selected = max(results, key=lambda result: (result["f1_score"], result["accuracy"]))
    best_name = selected["name"]
    final_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", model_candidates()[best_name]),
    ])
    final_pipeline.fit(features, labels)
    artifact = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "algorithm": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "classes": sorted(labels.unique().tolist()),
        "target_column": TARGET_COLUMN,
        "metrics": selected,
        "comparison": results,
        "test_samples": int(len(y_test)),
        "training_rows": int(len(frame)),
        "dataset_name": dataset_name or source.name,
        "synthetic_demo": "demo" in (dataset_name or source.name).lower(),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SecureMailScope demo data or train and compare classification models."
    )
    parser.add_argument("--dataset", type=Path, help="CSV to train from; omit to generate the labeled demo dataset.")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "securemailscope.joblib")
    parser.add_argument("--rows", type=int, default=480, help="Number of synthetic rows generated when --dataset is omitted.")
    args = parser.parse_args()
    DATASETS_DIR.mkdir(exist_ok=True)
    if args.dataset:
        source = args.dataset
    else:
        source = DATASETS_DIR / "email_security_demo.csv"
        generate_sample_dataset(source, n_samples=args.rows)
        print(f"Generated {args.rows} synthetic demonstration records: {source}")
    artifact = train_and_save(source, args.output, dataset_name=source.name)
    print(f"Selected model: {artifact['model_name']}")
    print(f"Saved model: {args.output}")
    for result in artifact["comparison"]:
        print(
            f"{result['name']:<22} accuracy={result['accuracy']:.3f} "
            f"precision={result['precision']:.3f} recall={result['recall']:.3f} f1={result['f1_score']:.3f}"
        )


if __name__ == "__main__":
    main()
