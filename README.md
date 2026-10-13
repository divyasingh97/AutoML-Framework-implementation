# AutoML Framework 🤖

An interactive **Automated Machine Learning (AutoML) Framework** built with Python and Streamlit that simplifies the process of training, comparing, and evaluating machine learning models without requiring users to write model-training code manually.

The application supports classification and regression tasks, automatic preprocessing, model comparison, performance visualization, and predictions through a user-friendly web interface.

## 🚀 Features

- **Upload your own dataset:** Upload CSV files to train machine learning models on your data.
- **Demo dataset:** Explore the framework using the IBM HR Analytics Employee Attrition dataset.
- **Classification and regression:** Choose the problem type that matches your dataset.
- **Automatic preprocessing:** Handles numerical and categorical features using appropriate preprocessing pipelines.
- **Multiple ML models:** Train and compare several machine learning algorithms.
- **Model evaluation:** View performance metrics and compare model results.
- **Visualizations:** Explore evaluation charts, confusion matrices, and classification reports.
- **Predictions:** View test-set predictions and make predictions for new records.
- **Download results:** Export predictions and the trained model.
- **Interactive interface:** Configure model training through Streamlit.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub

## 🧠 Machine Learning Algorithms

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

## 📊 Model Evaluation

The framework evaluates models using metrics appropriate to the selected problem.

**Classification metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

**Regression metrics**

The available regression evaluation metrics depend on the implementation in `app.py`. Check the app's displayed results before documenting specific metrics here.

## 📁 Project Structure

```text
AutoML-Framework-implementation/
│
├── app.py
├── sample_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

The IBM HR Analytics dataset is stored locally in the `data/` directory and excluded from GitHub.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/divyasingh97/AutoML-Framework-implementation.git
```

### 2. Open the project folder

```bash
cd AutoML-Framework-implementation
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows Git Bash:

```bash
source .venv/Scripts/activate
```

For Windows Command Prompt:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## 📂 Using the Framework

1. Launch the Streamlit application.
2. Select the dataset source.
3. Upload a CSV file or select the supported demo dataset.
4. Choose classification or regression.
5. Select the test-data percentage.
6. Train and compare the available models.
7. Review the evaluation metrics and visualizations.
8. Explore predictions and download available results.

**Dataset note:** The demo uses the IBM HR Analytics Employee Attrition dataset. To use that demo on another computer, obtain the dataset separately and place `WA_Fn-UseC_-HR-Employee-Attrition.csv` inside the `data/` directory. Alternatively, upload your own suitable CSV dataset.

## 🎯 Project Objective

The objective of this project is to make machine learning experimentation more accessible by bringing dataset input, preprocessing, model training, evaluation, and prediction into one interactive application.

## 🔮 Future Improvements

- Add more machine learning algorithms.
- Implement automated hyperparameter tuning.
- Add cross-validation for more reliable model evaluation.
- Support additional evaluation metrics and visualizations.
- Improve handling of missing values and imbalanced datasets.
- Deploy the application online for public access.

## 👩‍💻 Author

**Divya Jadon**  
BCA — Artificial Intelligence and Machine Learning  
Galgotias University

GitHub: [@divyasingh97](https://github.com/divyasingh97)

---

*This project was developed as a practical machine learning application to explore model training, evaluation, and deployment using Python and Streamlit.*
