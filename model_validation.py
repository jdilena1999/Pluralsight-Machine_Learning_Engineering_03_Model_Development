import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np

df = pd.read_csv(f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",header=None,skipinitialspace=True,na_values=" ?",
                 names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']).dropna()


# X Columns - All Columns Except Income
X = df.drop("income",axis=1)

#Y Column - Income => greater than 50K or less than/equal to 50K
y = df['income'].str.contains(">50K").astype(int) # 1 if it contains >50K, 0 otherwise

'''cat_columns = X.select_dtypes(include='object').columns.to_list()
num_columns = X.select_dtypes(exclude='object').columns.to_list()

for c in cat_columns:
    X[c].str.strip()'''
    
print(f"Dataset ready - {len(X):,} rows, class balance: {y.mean():.2%} high-income\n")

# 2. Load the pre-trained models
FILES = {
    "LogReg": "LogisticRegression.joblib",
    "RandForest":"RandomForest.joblib",
    "MLP": "MLPClassifier.joblib"
}


models = {}
for name,path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find '{path}' in working directory")
    models.setdefault(name,joblib.load(path))
    print(f"Loaded {name} pipeline from {path}")


# 3. Validation Strategies
cv_splits = {
    "StratifiedK5":StratifiedKFold(n_splits=5,shuffle=True,random_state=0),
    "KFold5":KFold(n_splits=5,shuffle=True,random_state=0),
    "TimeSeries3":TimeSeriesSplit(n_splits=3,test_size=2_000)
}

def blocked_progressive(X_df:pd.DataFrame, y_ser:pd.Series, pipe:Pipeline,block=2000) -> float:
    scores,start = [],0
    total = len(X_df)
    while start + 2 * block <= total:
        end_train = start + block               # Beginning: start - 0, block - 2_000 => 2_000
        end_test = end_train + block            # Beginning: end_train - 2_000, block - 2_000 => 4_000
        X_tr, y_tr = X_df.iloc[start:end_train],y_ser.iloc[start:end_train]         # Beginning: train on indices 0:2_000
        X_te, y_te = X_df.iloc[end_train:end_test],y_ser.iloc[end_train:end_test]   # Beginning: test on indices 2_000:4_000
        pipe.fit(X_tr,y_tr)
        scores.append(roc_auc_score(y_te,pipe.predict_proba(X_te)[:,1]))
        if len(scores) > 2 and abs(scores[-1] - scores[-2]) < 0.0001:               # Scores didn't get any better
            break
        start+=block
    return sum(scores) / len(scores)            # AUC

# 4. Run Benchmarks

rows = []

for m_name,pipe in models.items():
    # Start with Standard Cross-Validation options
    for s_name,splitter in cv_splits.items():
        t0 = time.perf_counter()
        auc = cross_val_score(pipe,X,y,cv=splitter,scoring='roc_auc',n_jobs=1).mean()
        duration = time.perf_counter() - t0
        rows.append((m_name,s_name,round(auc,4),round(duration,2)))
        
    # Now use Blocked Progressive
    t0 = time.perf_counter()
    auc = blocked_progressive(X,y,pipe)
    duration = time.perf_counter()-t0
    rows.append((m_name,"BlockedProg",round(auc,4),round(duration,2)))
    
df_results = pd.DataFrame(rows,columns=["Model","Strategy",'AUC','Seconds'])
df_pivot = df_results.pivot(index='Model',columns='Strategy')

df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
df_pivot = df_pivot.reset_index()

with pd.option_context("display.width",None):
    print("\n AUC and Runtime (sec) by Model and Strategy \n")
    print(df_pivot)