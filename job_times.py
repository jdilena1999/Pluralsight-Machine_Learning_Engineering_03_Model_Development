import datetime as dt    
import time  
import joblib

models = {
    "Logistic Regression":"LogisticRegression.joblib",
    "Random Forest":"RandomForest.joblib",
    "MLP Classifier": "MLPClassifier.joblib"
}
print(f"\n=== Cold-Start Deserialization Times ===\n")
for name,path in models.items():
    #start = dt.datetime.now()
    start = time.perf_counter()
    _ = joblib.load(path)
    duration = time.perf_counter() - start
    print(f"{name}: {duration:.4f} seconds")