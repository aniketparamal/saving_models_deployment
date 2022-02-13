import joblib
load_model=joblib.load('dib_79.pkl')
pred=load_model.predict([[10,20,50,70,40,60,30,30]])
print(pred)