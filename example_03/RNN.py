import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

layer_start = 5
hidden_layers = [5,5,5]
layer_end = 1

n_layers= 4

const_steps = 3000
const_learningRate = 0.0001
const_folds = 10

rate_training = []
rate_test = []
mean_rate_training = 0
mean_rate_test = 0
standar_desviation = 0

print("Jean Pierre Soto Chirinos")
print("Redes neuronales \n")


print("Datos de Test")
MLP = MLPClassifier(activation='relu', solver='sgd',
                    hidden_layer_sizes=(hidden_layers),
                    learning_rate_init = const_learningRate,
                    max_iter = const_steps)

for i in tqdm(range(1, const_folds+1)):
    
    print('-------------------------------------------------------')
    print('phoneme-10-'+str(i)+'\n')
    bD_train = pd.read_csv('./data/phoneme/phoneme-10-'+str(i)+'tra.csv')
    X_train = bD_train.loc[:,bD_train.columns !='cl']
    Y_train = bD_train.cl

    bD_test = pd.read_csv('./data/phoneme/phoneme-10-'+str(i)+'tst.csv')
    X_test = bD_test.loc[:,bD_test.columns !='cl']
    Y_test = bD_test.cl

    print('Cantidad del conjunto de entrenamiento: ',len(bD_train.cl),'\n')
    print('Cantidad del conjunto de prueba: ',len(bD_test.cl),'\n')
    #print(X_train,Y_train)
    for col_name in bD_train.columns: 
        if(col_name!='cl'):
            print(col_name,' caracteristica')
        else:
            print(col_name,' clase\n')

    print("Parametros: ",MLP.fit(X_train,Y_train))


    print("Cantidad de capas ocultas: ",len(hidden_layers))
    print("Capa 1 (entrada): ",layer_start)
    for i in range(0,len(hidden_layers)):
        print("Capa",i+2,': ',hidden_layers[i])

    print("Capa "+str(len(hidden_layers)+2)+" (salida): ",layer_end,'\n')
    print("Coeficiente de aprendizaje: ", const_learningRate)

    predictions_train = MLP.predict(X_train)
    print('Tasa de clasificación (training set)',accuracy_score(predictions_train,Y_train))
    rate_training.append(accuracy_score(predictions_train,Y_train))
    mean_rate_training += accuracy_score(predictions_train,Y_train)/const_folds

    predictions_test = MLP.predict(X_test)
    rate_test.append(accuracy_score(predictions_test,Y_test))
    mean_rate_test += accuracy_score(predictions_test,Y_test)/const_folds
    print('Tasa de clasificación (test set)',accuracy_score(predictions_test,Y_test),'\n\n')

print('-------------------------------------------------------\n')

for i in range(0, const_folds):
    print('Fold: ',i,' Rate Training: ',rate_training[i],'Rate Test: ',rate_test[i] )
print('Media Rate Training: ',mean_rate_training,' Media Rate Training: ',mean_rate_test)

sum_sd_train = 0
sum_sd_test = 0

for i in range(0, const_folds):
   sum_sd_train += math.pow((rate_training[i]-mean_rate_training),2)/(const_folds-1)
   sum_sd_test += math.pow((rate_test[i]-mean_rate_test),2)/(const_folds-1)

standar_desviation_train=math.sqrt(sum_sd_train)
standar_desviation_test=math.sqrt(sum_sd_test)

print('Desviación estandar training: ',standar_desviation_train,'Desviación estandar test: ',standar_desviation_test)