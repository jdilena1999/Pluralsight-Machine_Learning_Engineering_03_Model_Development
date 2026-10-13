import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression,LinearRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

## Training Models and Saving them to their own .joblib file


print(os.getcwd())
def greater_than_50k(x):
    return x.strip() == ">50K"

df = pd.read_csv(f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",header=None,
                 names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])

# Replace ? with NA values
df.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
# Drop NA values
df.dropna(inplace=True)

# X Columns - All Columns Except Income
X = df.drop("income",axis=1)

#Y Column - Income => greater than 50K or less than/equal to 50K
y = df['income'].apply(lambda x: x.strip() == '>50K')

cat_columns = X.select_dtypes(include='object').columns.to_list()
num_columns = X.select_dtypes(exclude='object').columns.to_list()

for c in cat_columns:
    X[c].str.strip()
# Split data into train and test
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2)



cat_pipeline = Pipeline([
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('scale',StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        #('strip',strip_transformer,cat_columns),
        ('cat',cat_pipeline,cat_columns),
        ('num',num_pipeline,num_columns)
    ]
)


models = {
    "LogisticRegression":LogisticRegression(max_iter=2000,solver='lbfgs',n_jobs=-1),
    "RandomForest":RandomForestClassifier(n_estimators=200,n_jobs=-1),
    "MLPClassifier":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=400)
}


results = {}

for name,model in models.items():
    print(f"Model: {name}") 
    pipeline = make_pipeline(preprocessor,model)
    
    # Fit model
    start_time = time.perf_counter()
    pipeline.fit(x_train,y_train)
    fit_time = time.perf_counter()-start_time
    
    # Save model to disk
    filename = f"{name}.joblib"
    joblib.dump(pipeline,filename)
    model_size_kb = os.path.getsize(filename) / 1024
    
    # Inference timing (single-row prediction - 1,000 row sample)
    start_time = time.perf_counter()
    for _ in range(1_000):
        pipeline.predict(x_test.iloc[[0]]) # Correct 2D input
    latency_ms = (time.perf_counter() - start_time)/1_000 * 1_000
    
    # Accuracy
    y_pred = pipeline.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    results[name] = {
        "Train Time (s)":fit_time,
        "Latency (ms)": latency_ms,
        "Model Size (KB)":round(model_size_kb,1),
        "Model Accuracy":round(acc,5)
        
    }
    
print(pd.DataFrame(results).transpose())
