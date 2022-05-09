import pickle 
import numpy as np
import pandas
#add a comment
#add some code
loaded_model = pickle.load(open('Models/GradientBoostingModel.pkl', 'rb'))

def predict(dropperc, mins, consecmonths, age, income):
    features=[]
    features.append(dropperc)
    features.append(mins)
    features.append(consecmonths)
    features.append(age)
    features.append(income)
    
    final = np.reshape(features, (1, -1))
    return ("Model predicts {} with a prediction probability of {}".format(loaded_model.predict(final)[0], np.around(loaded_model.predict_proba(final)[:,loaded_model.predict(final)][0][0], 4)))