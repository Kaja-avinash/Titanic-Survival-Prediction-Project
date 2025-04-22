# Titanic-Survival-Prediction-Project
# Titanic Survival Prediction

This project predicts passenger survival on the Titanic using machine learning. It includes data preprocessing, exploratory analysis, model training, and evaluation.

## Project Structure
titanic-survival-prediction/
│
├── data/
│   ├── raw/               # Original data files
│   │   └── tested.csv
│   └── processed/        # Processed data files
│
├── notebooks/
│   └── titanic_analysis.ipynb  # Jupyter notebook with full analysis
│
├── src/
│   ├── preprocessing.py   # Data cleaning and preprocessing
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation script
│   └── visualize.py       # Visualization scripts
│
├── models/
│   └── titanic_model.pkl  # Trained model file
│
├── reports/
│   ├── figures/           # Generated visualizations
│   └── metrics.txt        # Performance metrics
│
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

## Dataset

The dataset comes from [Kaggle](https://www.kaggle.com/competitions/titanic/data) and contains information about Titanic passengers.

Features:
- Survived: Target variable (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Sex: Gender
- Age: Age in years
- SibSp: # of siblings/spouses aboard
- Parch: # of parents/children aboard
- Fare: Passenger fare
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Preprocessing Steps

1. **Data Cleaning**:
   - Dropped irrelevant columns (Name, Cabin, Ticket, PassengerId)
   - Handled missing values:
     - Age: Filled with median
     - Embarked: Filled with mode
     - Fare: Filled with median

2. **Feature Engineering**:
   - Label encoded categorical variables (Sex, Embarked)
   - Standard scaled numerical features (Age, Fare)

## Model Selection

Used Random Forest Classifier with the following parameters:
- n_estimators = 100
- random_state = 42

## Performance Analysis

### Evaluation Metrics

| Metric     | Score   |
|------------|---------|
| Accuracy   | {value} |
| Precision  | {value} |
| Recall     | {value} |
| F1 Score   | {value} |
| ROC AUC    | {value} |



## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebook or individual scripts:
   - `python src/Titanic Survival Prediction.ipynb`

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle
