import numpy as np
import pandas as pd

def load_rice():
    df = pd.read_csv('data/rice.csv')
    X = df.drop('label', axis=1).values
    y = (df['label'] == 'Cammeo').astype(int).values
    return X, y

def load_parkinsons():
    df = pd.read_csv('data/parkinsons.csv')
    X = df.drop('Diagnosis', axis=1).values
    y = df['Diagnosis'].values
    return X, y

def load_credit():
    df = pd.read_csv('data/credit_approval.csv')
    # One-hot encode categorical columns
    cat_cols = [c for c in df.columns if 'cat' in c]
    df = pd.get_dummies(df, columns=cat_cols)  # pd.get_dummies is fine (not sklearn)
    X = df.drop('label', axis=1).values.astype(float)
    y = df['label'].values
    return X, y

def load_digits():
    from sklearn import datasets  # loading only — allowed
    digits = datasets.load_digits(return_X_y=True)
    return digits[0], digits[1]

def normalize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma