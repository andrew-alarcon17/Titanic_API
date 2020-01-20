import requests

url = 'http://localhost:5000/predict_api'
r = request.post(url, json={'Pclass':3, 'Sex':0, 'Age':5, 'SibSp':4, 'Fare':90, 'Embarked':2})

print(r.json())