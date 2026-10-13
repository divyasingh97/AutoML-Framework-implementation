
# 🤖 AutoML Framework

An interactive web application that automates basic machine learning workflows. Users can upload a CSV dataset, select a target column, train multiple models, compare their performance, and download the best trained model.

## ✨ Features

- Upload and preview CSV datasets
- Select Classification or Regression tasks
- Automatically preprocess numerical and categorical features
- Handle missing feature values
- Train and compare multiple machine learning models
- Evaluate models using standard performance metrics
- Visualize model performance with charts
- View actual and predicted values
- Download the selected best model
- Download test predictions as a CSV file

## 🧠 Machine Learning Models

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier

### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

## 📊 Evaluation Metrics

### Classification
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The framework selects the best classification model using weighted F1 score and the best regression model using the lowest RMSE.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 📁 Project Structure

```text
AutoML-Framework/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_data.csv
```

The `.venv/` virtual environment is excluded from Git using `.gitignore`.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AutoML-Framework.git
cd AutoML-Framework
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows Git Bash:**

```bash
source .venv/Scripts/activate
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload a CSV dataset.
3. Select Classification or Regression.
4. Choose the target column you want to predict.
5. Click **Train Models**.
6. Compare model metrics and the performance chart.
7. Review test predictions.
8. Download the best trained model or prediction results.

## ⚠️ Limitations

- The framework is designed for structured, tabular CSV datasets.
- Model performance depends on the dataset, target column, and available samples.
- Very small datasets can produce unreliable evaluation results.
- The current application does not automatically tune model hyperparameters.
- Downloaded models should be validated before real-world use.

## 👩‍💻 Author

**Divya Jadon**  
BCA — Artificial Intelligence and Machine Learning

## 📄 License

This project is available for educational and learning purposes.
