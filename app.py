
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoML Framework",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML Framework")
st.write(
    "Upload a CSV dataset, choose the target column, "
    "and automatically train and compare machine learning models."
)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    task = st.selectbox(
        "Select problem type",
        ["Classification", "Regression"],
    )

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------


with st.sidebar:
    st.header("⚙️ Configuration")

    dataset_mode = st.selectbox(
        "Choose dataset source",
        [
            "Upload Your Own CSV",
            "IBM HR Analytics Demo",
        ],
        key="dataset_mode_selectbox",
    )

    task = st.selectbox(
        "Select problem type",
        ["Classification", "Regression"],
        key="task_selectbox",
    )

    test_size = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="test_size_slider",
    )

# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error(
        "Your dataset must contain at least one row "
        "and two columns."
    )
    st.stop()

# Remove completely empty rows and columns.
df = df.dropna(how="all").dropna(axis=1, how="all")

if df.empty or len(df.columns) < 2:
    st.error("The dataset does not contain enough usable data.")
    st.stop()

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Total Columns", f"{len(df.columns):,}")
col3.metric(
    "Missing Values",
    f"{int(df.isna().sum().sum()):,}",
)


# --------------------------------------------------
# SELECT TARGET COLUMN
# --------------------------------------------------

target = st.selectbox(
    "Choose the target column to predict",
    options=list(df.columns),
)

data = df.dropna(subset=[target]).copy()

if len(data) < 5:
    st.error("At least five usable rows are required.")
    st.stop()

X = data.drop(columns=[target])
y = data[target]

if X.shape[1] == 0:
    st.error("Your dataset needs feature columns besides the target.")
    st.stop()

# Prepare the target.
if task == "Classification":
    y = y.astype(str)

    if y.nunique() < 2:
        st.error(
            "Classification requires at least two different target classes."
        )
        st.stop()

    if y.nunique() > 50:
        st.warning(
            "The target has more than 50 classes. "
            "Check whether Classification is appropriate."
        )

else:
    y = pd.to_numeric(y, errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(y) < 5:
        st.error(
            "Regression requires at least five rows "
            "with numeric target values."
        )
        st.stop()

# Reset indexes to keep features and targets aligned.
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Remove columns that contain no feature values.
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    st.error("No usable feature columns remain.")
    st.stop()

# Remove unsupported complex values and infinity from numeric columns.
for column in X.select_dtypes(include=["number"]).columns:
    X[column] = X[column].replace([np.inf, -np.inf], np.nan)

# Identify numeric and categorical features.
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# --------------------------------------------------
# AUTOMATIC PREPROCESSING
# --------------------------------------------------

transformers = []

if numeric_columns:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


# --------------------------------------------------
# SELECT MACHINE LEARNING MODELS
# --------------------------------------------------

if task == "Classification":
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

else:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }


# --------------------------------------------------
# TRAIN AND COMPARE MODELS
# --------------------------------------------------

st.subheader("🧠 Train and Compare Models")

if st.button("🚀 Train Models", type="primary"):

    try:
        # Stratification helps preserve class proportions when possible.
        stratify = None

        if task == "Classification":
            class_counts = y.value_counts()

            if (
                y.nunique() >= 2
                and class_counts.min() >= 2
                and int(np.ceil(len(y) * test_size / 100))
                >= y.nunique()
            ):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=stratify,
        )

        results = []
        fitted_models = {}
        predictions_by_model = {}

        progress = st.progress(0)
        status = st.empty()

        for index, (name, model) in enumerate(models.items()):

            status.write(f"Training {name}...")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            # Predict with this specific model.
            model_predictions = pipeline.predict(X_test)

            fitted_models[name] = pipeline
            predictions_by_model[name] = model_predictions

            # Evaluate classification models.
            if task == "Classification":

                accuracy = accuracy_score(
                    y_test,
                    model_predictions,
                )

                precision = precision_score(
                    y_test,
                    model_predictions,
                    average="weighted",
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    model_predictions,
                    average="weighted",
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    model_predictions,
                    average="weighted",
                    zero_division=0,
                )

                results.append(
                    {
                        "Model": name,
                        "Accuracy": accuracy,
                        "Precision (weighted)": precision,
                        "Recall (weighted)": recall,
                        "F1 Score (weighted)": f1,
                    }
                )

            # Evaluate regression models.
            else:
                mae = mean_absolute_error(
                    y_test,
                    model_predictions,
                )

                mse = mean_squared_error(
                    y_test,
                    model_predictions,
                )

                rmse = float(np.sqrt(mse))

                r2 = (
                    r2_score(y_test, model_predictions)
                    if len(y_test) >= 2
                    else np.nan
                )

                results.append(
                    {
                        "Model": name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2 Score": r2,
                    }
                )

            progress.progress(
                (index + 1) / len(models)
            )

        results_df = pd.DataFrame(results)

        # Select the best model using a suitable metric.
        if task == "Classification":
            ranking_metric = "F1 Score (weighted)"

            results_df = results_df.sort_values(
                ranking_metric,
                ascending=False,
            )

        else:
            ranking_metric = "RMSE"

            results_df = results_df.sort_values(
                ranking_metric,
                ascending=True,
            )

        best_name = results_df.iloc[0]["Model"]

        # IMPORTANT: Save the best model's predictions,
        # not the predictions from the last loop iteration.
        best_predictions = predictions_by_model[best_name]

        predictions_df = pd.DataFrame(
            {
                "Actual": y_test.to_numpy(),
                "Predicted": best_predictions,
            }
        )

        # Save results for Streamlit reruns.
        st.session_state["results_df"] = results_df
        st.session_state["fitted_models"] = fitted_models
        st.session_state["best_name"] = best_name
        st.session_state["predictions_df"] = predictions_df
        st.session_state["y_test"] = y_test
        st.session_state["X_test"] = X_test
        st.session_state["task"] = task
        st.session_state["target"] = target
        st.session_state["signature"] = (
            tuple(df.columns),
            target,
            task,
            test_size,
            len(df),
        )

        status.success("All models trained successfully!")

    except Exception as error:
        st.error(f"Training failed: {error}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

current_signature = (
    tuple(df.columns),
    target,
    task,
    test_size,
    len(df),
)

if (
    "results_df" in st.session_state
    and st.session_state.get("signature") == current_signature
):

    results_df = st.session_state["results_df"]
    fitted_models = st.session_state["fitted_models"]
    best_name = st.session_state["best_name"]
    predictions_df = st.session_state["predictions_df"]

    st.subheader("🏆 Model Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    if task == "Classification":
        st.success(
            f"Best model by weighted F1 score: {best_name}"
        )
    #python
    # Confusion matrix and classification report
    if task == "Classification":
        st.subheader("🔍 Confusion Matrix")

        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]

        best_pipeline = fitted_models[best_name]
        best_predictions = best_pipeline.predict(X_test)

        labels = best_pipeline.named_steps["model"].classes_

        cm = confusion_matrix(
            y_test,
            best_predictions,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )

        st.dataframe(cm_df, use_container_width=True)

        st.subheader("📋 Classification Report")

        report = classification_report(
            y_test,
            best_predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.round(3),
            use_container_width=True,
        )

    else:
        st.success(
            f"Best model by lowest RMSE: {best_name}"
        )

    # Display a warning if all scores are zero.
    metric_columns = [
        column
        for column in results_df.columns
        if column != "Model"
    ]

    if (
        task == "Classification"
        and (results_df[metric_columns] == 0).all().all()
    ):
        st.warning(
            "All classification metrics are zero. "
            "Check your dataset, target column, and predictions."
        )

    # Performance chart.
    st.subheader("📈 Model Performance")

    selected_metric = st.selectbox(
        "Choose a metric to visualize",
        metric_columns,
    )

    ascending = selected_metric in ["RMSE", "MAE"]

    chart_data = results_df.sort_values(
        selected_metric,
        ascending=ascending,
        na_position="last",
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        chart_data["Model"],
        chart_data[selected_metric],
    )

    ax.set_title(f"Model Comparison — {selected_metric}")
    ax.set_ylabel(selected_metric)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Display test predictions from the best model.
    st.subheader("🔮 Test Predictions")

    st.dataframe(
        predictions_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download the best trained model.
    model_bytes = io.BytesIO()

    joblib.dump(
        fitted_models[best_name],
        model_bytes,
    )

    st.download_button(
        label="⬇️ Download Best Model",
        data=model_bytes.getvalue(),
        file_name="best_automl_model.joblib",
        mime="application/octet-stream",
    )

    # Download predictions.
    csv_bytes = predictions_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Test Predictions",
        data=csv_bytes,
        file_name="test_predictions.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AutoML Framework | Built with Streamlit, Pandas, "
    "NumPy and Scikit-learn | Educational project"
)
