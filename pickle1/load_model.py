import pickle
loaded_model=pickle.load(open('dib_79.pkl','rb'))
pred=loaded_model.predict([[10,20,30,50,10,80,20,50]])
print(pred)