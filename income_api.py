from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
import onnxruntime as ort 
import joblib, os, re  

app = FastAPI(title="Income Prediction API")

# --- load model -----------------------------------------------------------------------------

USE_ONNX = os.getenv("USE_ONNX","1") == "1"
#print(f"USE_ONNX: {USE_ONNX}")
if USE_ONNX:
    sess = ort.InferenceSession("logreg.onnx",providers=["CPUExecutionProvider"])
    ONNX_INPUTS = [i.name for i in sess.get_inputs()]
else:
    pipe = joblib.load("LogisticRegression.joblib")
    ONNX_INPUTS = []

    
    
def sanitise(col:str) -> str:
    """ 
    Apply the exact rules skl2onnx uses:
    - replace any char other than [A-Za-z0-9] with '_'
    - collapse multiple '_' into one
    - strip leading/trailing '_'
    
    """
    col = re.sub(r"[^0-9A-Za-z_]",'_',col)
    col = re.sub(r"__+","_",col)
    return col.strip("_")

SANITISED_LOOKUP = {c:sanitise(c) for c in [
    "age",'workclass','fnlwgt','education','education-num',
    'marital-status','occupation','relationship','race','sex',
    'capital-gain','capital-loss','hours-per-week','native-country'
]}

class Record(BaseModel):
    data:dict           # original column names
    
    # --- endpoint -------------------------------------------------------------------------------
    @app.post("/predict")
    def predict(rec:Record):
        # Build the dict ONNX expects
        input_dict = {}
        for raw_key,val in rec.data.items():
            onnx_key = SANITISED_LOOKUP.get(raw_key)
            #print(f"raw key: {raw_key}\nonnx_key: {onnx_key}")
            if onnx_key is None:
                raise ValueError(f"Unexpected Key: {raw_key}")
            if not isinstance(val,(int,float)):
                val = str(val).strip()
                val = ' '+val
            arr = np.array([[val]],dtype=np.float32 if isinstance(val,(int,float)) else np.str_)
            #print(f"arr: {arr}")
            input_dict.setdefault(onnx_key, arr)    
        missing = set(ONNX_INPUTS) - input_dict.keys()      # ONNX_INPUTS that are not input_dict keys just created
        if missing:
            raise ValueError(f"Missing Keys for ONNX: {missing}")
        
        # run inference
        if USE_ONNX:
            #print(f"Using ONNX")
            proba = float(sess.run(None,input_dict)[1][0][1])   # class 1 probability (> 50K)
        else:
            #print(f"Using SKLearn Pipeline")
            import pandas as pd
            for key,val in rec.data.items():
                if not isinstance(val,(int,float)):
                    val = str(val).strip()
                    #val = ' '+val
                    print(f"Key: {key} - {val}")
            row = pd.DataFrame([rec.data])
            proba = float(pipe.predict_proba(row)[0][1])
        return {"Probability > 50K":proba}
    
if __name__ == '__main__':
    test_record = {'age':39
                ,'workclass':'State-gov'
                ,'fnlwgt': 77516
                ,'education':'Bachelors'
                ,'education-num':13
                ,'marital-status':'Never-married'
                ,'occupation': 'Adm-clerical'
                ,'relationship':'Not-in-family'
                ,'race':'White'
                ,'sex':'Male'
                ,'capital-gain':2174
                ,'capital-loss':0
                ,'hours-per-week':40
                ,'native-country':'United-States'
                    }
    test_rec = Record(data=test_record)
    




'''
# Docker File 20L, 434B

# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY logreg.onnx .

# Set Environment variable so app uses ONNX
ENV USE_ONNX=1

# Start the FastAPI app using Unicorn
CMD ["uvicorn", "main:app","--host","0.0.0.0","--port","8000"]



'''



'''
terminal

sudo docker build -t income_api .

sudo docker run -d -p 8000:8000 --name income income_api

'''


'''
Testing API in Terminal:

curl -s -X POST http://localhost:8000/predict

-H "Content-Type: application/json"

-d '{
    "data": {
    "age":39,
    "workclass":"State-gov",
    "fnlwgt":77516,
    "education":"Bachelors",
    "education-num":13,
    "marital-status":"Never married",
    "occupation":"Adm-clerical",
    "relationship":"Not-in-family",
    "race":"White",
    "sex": "Male",
    "capital-gain":2174,
    "capital-loss":0,
    "hours-per-week":40,
    "native-country": "United-States"
}
}'; echo

'''



'''
Degradation Modes of Deployed Models

# Latency Spikes
# Memory Blow-Up
# CPU Oversubscription/Noise
# Cold Start Drag

'''

'''
Locust Testing via terminal

sudo docker update --cpus="16" income




sudo docker update --cpus="1" income



locust --headless --users 100 --spawn-rate 10 --host http://localhost:8000 --run-time 60s




'''


'''
Handling Service Degradation

# Model Shunting
# Request Queuing
# Timeouts

'''



