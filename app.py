
from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AutoML Framework",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML Framework")
st.write(
    "Automatically train, compare, evaluate, and use machine learning models."
)


# ---------------- DATASET LOADER ----------------
def load_demo_dataset():
    paths = [
        Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        Path("WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    ]

    for path in paths:
        if path.exists():
            return pd.read_csv(path), str(path)

    return None, None


# ---------------- PREPROCESSING ----------------
def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        col for col in X.columns if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers)


# ---------------- MACHINE LEARNING MODELS ----------------
def get_models(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }


# ---------------- TRAIN AND COMPARE ----------------
def train_models(X, y, task, test_size):
    stratify_values = None

    if task == "Classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify_values = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_values,
    )

    results = []
    fitted_models = {}
    predictions = {}

    for model_name, estimator in get_models(task).items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)

        fitted_models[model_name] = pipeline
        predictions[model_name] = predicted

        if task == "Classification":
            results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision (weighted)": precision_score(
                    y_test, predicted,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall (weighted)": recall_score(
                    y_test, predicted,
                    average="weighted",
                    zero_division=0,
                ),
                "F1 Score (weighted)": f1_score(
                    y_test, predicted,
                    average="weighted",
                    zero_division=0,
                ),
            })
        else:
            results.append({
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predicted),
                "RMSE": np.sqrt(
                    mean_squared_error(y_test, predicted)
                ),
                "R² Score": r2_score(y_test, predicted),
            })

    results_df = pd.DataFrame(results)

    if task == "Classification":
        results_df = results_df.sort_values(
            "F1 Score (weighted)", ascending=False
        ).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(
            "RMSE", ascending=True
        ).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]

    predictions_df = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": np.asarray(predictions[best_name]),
    })

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    return {
        "results_df": results_df,
        "fitted_models": fitted_models,
        "best_name": best_name,
        "predictions_df": predictions_df,
        "X_test": X_test,
        "y_test": y_test,
        "input_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "training_X": X,
    }


# ---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    dataset_mode = st.radio(
        "Choose dataset source",
        ["Demo dataset", "Upload CSV"],
    )

    task = st.selectbox(
        "Choose task",
        ["Classification", "Regression"],
    )

    test_percentage = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )


# ---------------- LOAD DATA ----------------
df = None
dataset_label = ""

if dataset_mode == "Demo dataset":
    df, dataset_label = load_demo_dataset()

    if df is None:
        st.warning(
            "Demo dataset not found. Place the IBM HR CSV inside "
            "the data folder, or select Upload CSV."
        )
        st.code(
            "AutoML-Framework/\n"
            "├── app.py\n"
            "└── data/\n"
            "    └── WA_Fn-UseC_-HR-Employee-Attrition.csv"
        )
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            dataset_label = uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if df is None:
    if dataset_mode == "Upload CSV":
        st.info("Upload a CSV file to begin.")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.caption(
    f"Dataset: {dataset_label} | "
    f"{df.shape[0]} rows | {df.shape[1]} columns"
)
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")


# ---------------- SELECT TARGET ----------------
st.subheader("🎯 Select Target Column")

default_index = (
    df.columns.get_loc("Attrition")
    if "Attrition" in df.columns
    else len(df.columns) - 1
)

target = st.selectbox(
    "Choose the target column to predict",
    options=df.columns.tolist(),
    index=default_index,
)

signature = (
    dataset_mode,
    dataset_label,
    tuple(df.columns),
    target,
    task,
    test_percentage,
    len(df),
)

# ---------------- TRAIN BUTTON ----------------
if st.button("🚀 Train Models", type="primary"):
    try:
        if df.shape[1] < 2:
            raise ValueError(
                "The dataset needs at least one feature and one target."
            )

        with st.spinner("Training models. Please wait..."):
            working_df = df.dropna(subset=[target]).copy()

            X = working_df.drop(columns=[target])
            y = working_df[target]

            # Remove completely empty feature columns.
            empty_columns = [
                col for col in X.columns if X[col].isna().all()
            ]
            if empty_columns:
                X = X.drop(columns=empty_columns)

            if X.shape[1] == 0:
                raise ValueError("No usable feature columns were found.")

            if task == "Classification":
                if y.nunique() < 2:
                    raise ValueError(
                        "The target must contain at least two classes."
                    )
                y = y.astype(str)
            else:
                y = pd.to_numeric(y, errors="coerce")
                valid = y.notna()
                X = X.loc[valid]
                y = y.loc[valid]

                if len(y) < 2:
                    raise ValueError(
                        "Regression needs at least two numeric target values."
                    )

            trained = train_models(
                X,
                y,
                task,
                test_percentage / 100,
            )

            # Save results so they remain available after Streamlit reruns.
            st.session_state["trained_results"] = trained
            st.session_state["trained_signature"] = signature

    except Exception as exc:
        st.session_state.pop("trained_results", None)
        st.session_state.pop("trained_signature", None)
        st.error(f"Training failed: {exc}")


# ============================================================
# DISPLAY RESULTS AND PREDICTION FORM
# Keep the entire section inside this condition.
# ============================================================
if (
    "trained_results" in st.session_state
    and st.session_state.get("trained_signature") == signature
):
    trained = st.session_state["trained_results"]

    results_df = trained["results_df"]
    fitted_models = trained["fitted_models"]
    best_name = trained["best_name"]
    predictions_df = trained["predictions_df"]
    X_test = trained["X_test"]
    y_test = trained["y_test"]
    input_columns = trained["input_columns"]
    numeric_columns = trained["numeric_columns"]
    training_X = trained["training_X"]
    best_pipeline = fitted_models[best_name]

    st.success("All models trained successfully!")

    # ---------------- MODEL COMPARISON ----------------
    st.subheader("🏆 Model Comparison")

    numeric_metrics = results_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    st.dataframe(
        results_df.style.format({
            col: "{:.4f}" for col in numeric_metrics
        }),
        use_container_width=True,
        hide_index=True,
    )

    if task == "Classification":
        st.success(
            f"Best model by weighted F1 score: {best_name}"
        )
        chart_metric = "F1 Score (weighted)"
    else:
        st.success(f"Best model by lowest RMSE: {best_name}")
        chart_metric = "RMSE"

    # ---------------- PERFORMANCE CHART ----------------
    st.subheader("📊 Model Performance Chart")
    st.bar_chart(
        results_df.set_index("Model")[[chart_metric]]
    )

    # ---------------- EVALUATION ----------------
    if task == "Classification":
        st.subheader("🔵 Confusion Matrix")

        actual = pd.Series(y_test).astype(str)
        predicted = pd.Series(
            best_pipeline.predict(X_test)
        ).astype(str)

        labels = sorted(actual.unique().tolist())
        matrix = confusion_matrix(
            actual, predicted, labels=labels
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )

        st.dataframe(matrix_df, use_container_width=True)

        with st.expander("View Classification Report"):
            report = classification_report(
                y_test,
                best_pipeline.predict(X_test),
                zero_division=0,
                output_dict=True,
            )
            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True,
            )

    # ---------------- DOWNLOADS ----------------
    st.subheader("📥 Download Results")

    st.download_button(
        "Download best-model predictions (CSV)",
        data=predictions_df.to_csv(index=False).encode("utf-8"),
        file_name="automl_predictions.csv",
        mime="text/csv",
    )

    model_buffer = io.BytesIO()
    joblib.dump(best_pipeline, model_buffer)

    st.download_button(
        "Download best trained model",
        data=model_buffer.getvalue(),
        file_name="best_automl_model.joblib",
        mime="application/octet-stream",
    )

    # ========================================================
    # PREDICT ON NEW DATA
    # This is deliberately inside the results condition.
    # ========================================================
    st.divider()
    st.subheader("🧪 Predict on New Data")

    st.write(
        "Enter values for a new record. The best trained model "
        "will use these values to make a prediction."
    )
    st.caption(f"Using best model: {best_name}")

    with st.form("new_record_prediction_form"):
        new_record = {}
        form_columns = st.columns(2)

        for index, column in enumerate(input_columns):
            with form_columns[index % 2]:

                if column in numeric_columns:
                    median_value = pd.to_numeric(
                        training_X[column], errors="coerce"
                    ).median()

                    if (
                        pd.isna(median_value)
                        or not np.isfinite(median_value)
                    ):
                        median_value = 0.0

                    new_record[column] = st.number_input(
                        column,
                        value=float(median_value),
                        key=f"new_record_{column}",
                    )

                else:
                    choices = (
                        training_X[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if not choices:
                        choices = ["Unknown"]

                    new_record[column] = st.selectbox(
                        column,
                        options=choices,
                        key=f"new_record_{column}",
                    )

        predict_clicked = st.form_submit_button(
            "🔮 Predict",
            type="primary",
        )

    # ---------------- MAKE PREDICTION ----------------
    if predict_clicked:
        try:
            new_data = pd.DataFrame(
                [new_record],
                columns=input_columns,
            )

            for column in numeric_columns:
                new_data[column] = pd.to_numeric(
                    new_data[column], errors="coerce"
                )

            prediction = best_pipeline.predict(new_data)

            st.success("Prediction completed successfully!")
            st.subheader("Prediction Result")
            st.write(f"**Selected Model:** {best_name}")
            st.write(f"**Predicted Value:** {prediction[0]}")

            if (
                task == "Classification"
                and hasattr(best_pipeline, "predict_proba")
            ):
                probabilities = best_pipeline.predict_proba(new_data)[0]
                classes = best_pipeline.named_steps["model"].classes_

                probability_df = pd.DataFrame({
                    "Class": classes.astype(str),
                    "Probability": probabilities,
                }).sort_values(
                    "Probability", ascending=False
                )

                st.write("**Class probabilities**")
                st.dataframe(
                    probability_df.style.format({
                        "Probability": "{:.2%}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            prediction_csv = pd.DataFrame({
                "Predicted Value": [str(prediction[0])]
            }).to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download this prediction (CSV)",
                data=prediction_csv,
                file_name="new_record_prediction.csv",
                mime="text/csv",
                key="download_new_prediction",
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

else:
    st.info(
        "Choose your target column and click Train Models to display "
        "the model comparison and new-record prediction form."
    )
