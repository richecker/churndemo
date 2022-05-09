import pickle 
import numpy as np

print('loading in pickled model file')

loaded_model = pickle.load(open('Models/cvJobSelectedModel.pkl', 'rb'))
 
def predict(dropperc, mins, consecmonths, age, income):
    features=[]
    features.append(dropperc)
    features.append(mins)
    features.append(consecmonths)
    features.append(age)
    features.append(income)

    final = np.reshape(features, (1, -1))
    return ("Model predicts {} with a prediction probability of {}".\
            format(loaded_model.predict(final)[0], np.around(loaded_model.predict_proba(final)[:,loaded_model.predict(final)][0][0], 4)))