import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.set_page_config(page_title="Loan Modeling App", layout="wide")
st.title("Bank Personal Loan Modeling")

TARGET = "Personal Loan"


def evaluate_classifier(name, model, Xtr, Xte, ytr, yte, scaled=False):
    if scaled:
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
    else:
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    auc = roc_auc_score(yte, proba)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    }


def build_pipeline(raw_df):
    df = raw_df.copy()
    preprocessing_notes = []

    # 1) Basic Cleaning
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
        preprocessing_notes.append("Dropped column: ID")

    if "CCAvg" in df.columns:
        df["CCAvg_Annual"] = df["CCAvg"] * 12
        preprocessing_notes.append("Created CCAvg_Annual = CCAvg * 12")

    if "Zip Code" in df.columns:
        df["Zip Code"] = df["Zip Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        df["ZIP_Prefix_3"] = df["Zip Code"].str[:3]
        zip_freq = df["ZIP_Prefix_3"].value_counts(normalize=True)
        df["ZIP_Prefix_3_Freq"] = df["ZIP_Prefix_3"].map(zip_freq)
        df.drop(columns=["Zip Code"], inplace=True)
        preprocessing_notes.append("ZIP handling: created ZIP_Prefix_3 + ZIP_Prefix_3_Freq, dropped Zip Code")

    if "Experience" in df.columns:
        df["Experience"] = df["Experience"].apply(lambda x: abs(x) if pd.notna(x) else x)
        preprocessing_notes.append("Converted negative Experience values to absolute")

    df_eda = df.copy()

    # 2) EDA stats
    missing_count = df_eda.isna().sum()
    missing_pct = (df_eda.isna().mean() * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct}).sort_values(
        "missing_pct", ascending=False
    )

    if TARGET not in df_eda.columns:
        raise ValueError(f"Target column '{TARGET}' not found in uploaded dataset.")

    target_count = df_eda[TARGET].value_counts()
    target_pct = df_eda[TARGET].value_counts(normalize=True).mul(100).round(2)

    uni_cols = [c for c in ["Income", "CCAvg_Annual", "CCAvg", "Age", "Experience", "Mortgage"] if c in df_eda.columns]
    outlier_cols = [c for c in ["Income", "Mortgage", "CCAvg_Annual", "CCAvg"] if c in df_eda.columns]

    # 3) Preprocessing + Feature Engineering
    for c in df.columns:
        if df[c].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0])

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    low_card, high_card = [], []
    if obj_cols:
        low_card = [c for c in obj_cols if df[c].nunique() <= 10]
        high_card = [c for c in obj_cols if df[c].nunique() > 10]

        if high_card:
            for c in high_card:
                freq = df[c].value_counts(normalize=True)
                df[c + "_freq"] = df[c].map(freq)
            df.drop(columns=high_card, inplace=True)

        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=True)

    # 4) Train/Test Split
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Train Models
    models = {}
    models["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    models["SVM (RBF)"] = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced_subsample"
    )

    xgb_available = True
    try:
        from xgboost import XGBClassifier
    except Exception:
        xgb_available = False

    if xgb_available:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

    # 6) Evaluation + Comparison
    results = []
    results.append(evaluate_classifier("Logistic Regression", models["Logistic Regression"], X_train_scaled, X_test_scaled, y_train, y_test, scaled=True))
    results.append(evaluate_classifier("SVM (RBF)", models["SVM (RBF)"], X_train_scaled, X_test_scaled, y_train, y_test, scaled=True))
    results.append(evaluate_classifier("Random Forest", models["Random Forest"], X_train, X_test, y_train, y_test, scaled=False))
    if xgb_available:
        results.append(evaluate_classifier("XGBoost", models["XGBoost"], X_train, X_test, y_train, y_test, scaled=False))

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    best_model = models[best_model_name]

    if best_model_name in ["Logistic Regression", "SVM (RBF)"]:
        best_model.fit(X_train_scaled, y_train)
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)

    class_report = classification_report(y_test, y_pred_best)

    # 7) Feature Importance
    explain_model = None
    explain_name = None
    feature_importance = None

    if best_model_name in ["Random Forest", "XGBoost"]:
        explain_model = best_model
        explain_name = best_model_name
    else:
        explain_model = models["Random Forest"]
        explain_name = "Random Forest (for Feature Importance)"
        explain_model.fit(X_train, y_train)

    if hasattr(explain_model, "feature_importances_"):
        feature_importance = pd.Series(explain_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # 9) Clustering
    cluster_features = [c for c in ["Income", "CCAvg_Annual", "Family", "Education", "Mortgage", "CD Account"] if c in df.columns]
    cluster_counts = None
    cluster_profile = None
    if len(cluster_features) > 0:
        cluster_df = df[cluster_features].copy()
        cluster_scaler = StandardScaler()
        cluster_scaled = cluster_scaler.fit_transform(cluster_df)

        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_scaled)
        df["Cluster"] = clusters

        cluster_counts = df["Cluster"].value_counts()
        cluster_profile = df.groupby("Cluster")[cluster_features + [TARGET]].mean().round(2)

    return {
        "raw_df": raw_df,
        "df_eda": df_eda,
        "df_model": df,
        "preprocessing_notes": preprocessing_notes,
        "missing_df": missing_df,
        "target_count": target_count,
        "target_pct": target_pct,
        "uni_cols": uni_cols,
        "outlier_cols": outlier_cols,
        "obj_cols": obj_cols,
        "low_card": low_card,
        "high_card": high_card,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "models": models,
        "xgb_available": xgb_available,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "class_report": class_report,
        "feature_importance": feature_importance,
        "explain_name": explain_name,
        "cluster_counts": cluster_counts,
        "cluster_profile": cluster_profile
    }


def predict_customer_loan_acceptance(customer_dict, pipe):
    df_eda = pipe["df_eda"]
    X = pipe["X"]
    scaler = pipe["scaler"]
    best_model = pipe["best_model"]
    best_model_name = pipe["best_model_name"]

    row = pd.DataFrame([customer_dict])

    if "CCAvg_Annual" not in row.columns and "CCAvg" in row.columns:
        row["CCAvg_Annual"] = row["CCAvg"] * 12

    if "Zip Code" in row.columns:
        row["Zip Code"] = row["Zip Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        row["ZIP_Prefix_3"] = row["Zip Code"].str[:3]
        row.drop(columns=["Zip Code"], inplace=True)

    if "ZIP_Prefix_3" in row.columns and "ZIP_Prefix_3_Freq" not in row.columns:
        zip_freq = df_eda["ZIP_Prefix_3"].value_counts(normalize=True) if "ZIP_Prefix_3" in df_eda.columns else pd.Series(dtype=float)
        row["ZIP_Prefix_3_Freq"] = row["ZIP_Prefix_3"].map(zip_freq).fillna(0.0)
        row.drop(columns=["ZIP_Prefix_3"], inplace=True)

    for c in X.columns:
        if c not in row.columns:
            row[c] = 0

    row = row[X.columns]

    for c in row.columns:
        if row[c].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_eda[c]) if c in df_eda.columns else True:
                row[c] = row[c].fillna(df_eda[c].median() if c in df_eda.columns else 0)
            else:
                row[c] = row[c].fillna(df_eda[c].mode().iloc[0] if c in df_eda.columns else 0)

    if best_model_name in ["Logistic Regression", "SVM (RBF)"]:
        row_scaled = scaler.transform(row)
        prob = best_model.predict_proba(row_scaled)[0, 1]
        pred = int(prob >= 0.5)
    else:
        prob = best_model.predict_proba(row)[0, 1]
        pred = int(prob >= 0.5)

    return pred, float(prob)


# Sidebar: Upload + Modules
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

module = st.sidebar.radio(
    "Modules",
    ["Pre-processing", "EDA", "Model Training", "Evaluation", "Visualizations", "Predictions"]
)

if uploaded_file is None:
    st.info("Please upload a dataset from the left sidebar.")
    st.stop()

# Read uploaded file properly
file_name = uploaded_file.name.lower()
file_bytes = uploaded_file.getvalue()

if file_name.endswith(".csv"):
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
else:
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    default_idx = excel_file.sheet_names.index("Data") if "Data" in excel_file.sheet_names else 0
    selected_sheet = st.sidebar.selectbox("Select sheet", excel_file.sheet_names, index=default_idx)
    raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_sheet)

# Show data information immediately after upload
st.subheader("Loaded Data Information")
st.write(f"Loaded data shape: {raw_df.shape}")
st.dataframe(raw_df.head())

try:
    pipe = build_pipeline(raw_df)
except Exception as e:
    st.error(f"Error while processing dataset: {e}")
    st.stop()

if module == "Pre-processing":
    st.subheader("Pre-processing")
    for note in pipe["preprocessing_notes"]:
        st.write(f"- {note}")

    st.write("Object columns found:", pipe["obj_cols"])
    st.write("Low-cardinality object columns:", pipe["low_card"])
    st.write("High-cardinality object columns:", pipe["high_card"])
    st.write("Final model dataframe shape:", pipe["df_model"].shape)
    st.dataframe(pipe["df_model"].head())

elif module == "EDA":
    st.subheader("EDA")

    st.write("--- Missing Values ---")
    st.dataframe(pipe["missing_df"])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=pipe["missing_df"].index, y=pipe["missing_df"]["missing_pct"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_title("Missing Values (%) by Column")
    ax.set_ylabel("Missing %")
    plt.tight_layout()
    st.pyplot(fig)

    st.write("--- Target Balance ---")
    st.write(pipe["target_count"])
    st.write((pipe["target_pct"].astype(str) + "%"))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=pipe["df_eda"], ax=ax)
    ax.set_title("Target Balance: Personal Loan (0/1)")
    plt.tight_layout()
    st.pyplot(fig)

    for c in pipe["uni_cols"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(pipe["df_eda"][c], bins=30, kde=True, ax=ax)
        ax.set_title(f"Univariate Distribution: {c}")
        plt.tight_layout()
        st.pyplot(fig)

elif module == "Model Training":
    st.subheader("Model Training")
    st.write("Models trained:")
    for m in pipe["models"].keys():
        st.write(f"- {m}")

    if not pipe["xgb_available"]:
        st.info("XGBoost is not installed. To include XGBoost: pip install xgboost")

    st.write("--- Model Comparison (sorted by ROC_AUC) ---")
    st.dataframe(pipe["results_df"])
    st.success(f"Best model (by ROC_AUC): {pipe['best_model_name']}")

elif module == "Evaluation":
    st.subheader("Evaluation")
    st.dataframe(pipe["results_df"])

    if pipe["best_model_name"] in ["Logistic Regression", "SVM (RBF)"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_estimator(pipe["best_model"], pipe["X_test_scaled"], pipe["y_test"], ax=ax)
        ax.set_title(f"ROC Curve - Best Model: {pipe['best_model_name']}")
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_estimator(pipe["best_model"], pipe["X_test_scaled"], pipe["y_test"], ax=ax)
        ax.set_title(f"Confusion Matrix - Best Model: {pipe['best_model_name']}")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_estimator(pipe["best_model"], pipe["X_test"], pipe["y_test"], ax=ax)
        ax.set_title(f"ROC Curve - Best Model: {pipe['best_model_name']}")
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_estimator(pipe["best_model"], pipe["X_test"], pipe["y_test"], ax=ax)
        ax.set_title(f"Confusion Matrix - Best Model: {pipe['best_model_name']}")
        plt.tight_layout()
        st.pyplot(fig)

    st.text("Classification Report (Best Model):")
    st.code(pipe["class_report"])

elif module == "Visualizations":
    st.subheader("Visualizations")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pipe["results_df"]["Model"], pipe["results_df"]["ROC_AUC"])
    ax.set_title("Model Comparison by ROC-AUC")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticklabels(pipe["results_df"]["Model"], rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    if pipe["feature_importance"] is not None:
        st.write(f"Top Feature Importances ({pipe['explain_name']}):")
        st.dataframe(pipe["feature_importance"].head(12).to_frame("importance"))

        fig, ax = plt.subplots(figsize=(8, 5))
        pipe["feature_importance"].head(12).sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)

    if pipe["cluster_counts"] is not None:
        st.write("--- Customer Segmentation (KMeans) ---")
        st.write(pipe["cluster_counts"])

        st.write("Cluster Profile (means):")
        st.dataframe(pipe["cluster_profile"])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Cluster", data=pipe["df_model"], ax=ax)
        ax.set_title("Cluster Sizes (KMeans)")
        plt.tight_layout()
        st.pyplot(fig)

elif module == "Predictions":
    st.subheader("Predictions")
    st.write("Enter values manually for all model features (no default values).")

    user_inputs = {}
    cols = st.columns(3)
    feature_list = list(pipe["X"].columns)

    for i, feat in enumerate(feature_list):
        with cols[i % 3]:
            user_inputs[feat] = st.text_input(feat, value="", placeholder=f"Enter {feat}")

    if st.button("Predict Personal Loan"):
        missing_fields = [k for k, v in user_inputs.items() if str(v).strip() == ""]
        if missing_fields:
            st.error(f"Please enter all feature values. Missing: {', '.join(missing_fields[:8])}" + (" ..." if len(missing_fields) > 8 else ""))
        else:
            try:
                customer_dict = {k: float(v) for k, v in user_inputs.items()}
                pred, prob = predict_customer_loan_acceptance(customer_dict, pipe)
                st.success(f"Predicted Personal Loan (0/1): {pred}")
                st.success(f"Propensity Score (probability of acceptance): {round(prob, 4)}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
