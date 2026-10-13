from locust import HttpUser,task,between
import json
from income_api import Record
with open("payload.json",'r') as f:
    sample_payload = json.load(f)

class IncomePredictUser(HttpUser):
    wait_time = between(0.1,0.3)
    
    @task
    def predict_income(self):
        self.client.post("/predict",json=sample_payload)
        

'''if __name__ == '__main__':
    with open("payload.json",'r') as f:
        sample = json.load(f)
    
    pulls = []
    for k in sample.keys():
        pulls.append(sample[k])
    for p in pulls:
        rec = Record(data=p)
        print(rec.predict())'''