import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv

import math

# Definiendo constantes
FI = 2/(pow(5,1/2)-1) # constante FI
PI = math.pi # constante PI
# Crear numero aleatorios con Numpy y numeros Irracionales
# El conjunto de datos fue generado por las siguientes lineas de codigo, descomentar para probar
#arrnum_dataX = np.linspace(-100, 110, N) # conjunto de numeros aleatorios entre dos numeros
#arrnum_dataY = arrnum_dataX + np.random.randn(*arrnum_dataX.shape) * FI + PI # numero aleatorio a partir del primer conjunto

#conjunto de datos
#arrnum_dataX = [-100. , -70. , -40. , -10.  , 20.  , 50. ,  80. , 110.]
#arrnum_dataY =[-98.75652188 , -63.65906766 , -36.06312385  , -5.8395074 ,  21.507873,  53.55319652,  81.15010806, 115.37652695]
arrnum_dataX = []
arrnum_dataY = []

with open("100.csv",'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            arrnum_dataX.append(float(row[0]))
            arrnum_dataY.append(float(row[1]))
N = len(arrnum_dataX)

print(arrnum_dataX)
print(arrnum_dataY)

#Se plotea los puntos escogidos 

plt.plot(arrnum_dataX,arrnum_dataY,"ro",label="Datos de entrenamiento")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Puntos de entrenamiento')
plt.savefig("Puntos_grad_desc.png")
plt.show()

#Constantes
const_steps = 100000
const_learningRate = 0.0002
const_stopCriteria = 1e-8

#Variables
var_intercept = 0
var_slope = 0

# Ciclo para hallar los valores de los nuevos interceptos y pendientes

for step in tqdm(range(0, const_steps)):
    var_intercept_gradient = 0
    var_slope_gradient = 0
    for i in range(0, len(arrnum_dataX)):
        var_intercept_gradient -= (2/N) * (arrnum_dataY[i] - (var_intercept + var_slope * arrnum_dataX[i]))
        var_slope_gradient -= (2/N) * (arrnum_dataY[i] - (var_intercept + var_slope * arrnum_dataX[i])) * arrnum_dataX[i]
        
    var_intercept =  var_intercept - (const_learningRate * var_intercept_gradient)
    var_slope = var_slope - (const_learningRate * var_slope_gradient)
    print("\nIteración ", step+1)
    print("Intercepto gradiente (derivada): ",str(var_intercept_gradient)[0:8]," Pendiente gradiente (derivada): ",str(var_slope_gradient)[0:8])
    print("Intercepto anterior: ",str(var_intercept)[0:8]," Pendiente anterior: ",str(var_slope)[0:8])
    print("Criterio Intercepto (nuevo) ",str(abs(const_learningRate * var_intercept_gradient))[0:8])
    print("Criterio Pendiente (nuevo)",str(abs(const_learningRate * var_slope_gradient))[0:8])
    print("Criterio max",max(abs(const_learningRate * var_intercept_gradient), abs(const_learningRate * var_slope_gradient)))
    print("Coeficiente de aprendizaje: ", const_learningRate)
    

    if max(abs(const_learningRate * var_intercept_gradient), abs(const_learningRate * var_slope_gradient)) < const_stopCriteria:
        break

print("Los valores definitivos: ", " Intercepto: ", var_intercept," Pendiente: ",  var_slope, "en pasos", step+1)


start= [var_intercept+var_slope*arrnum_dataY[0],var_intercept+var_slope*arrnum_dataY[N-1]]
end= [arrnum_dataY[0],arrnum_dataY[N-1]]
plt.plot(end,start,label="Regresion")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Regresion Lineal')
plt.text(0.6, 0.6, r'$f(x) = (%s) + (%s)*Weight $'%(str(var_intercept)[0:6],str(var_slope)[0:6]), fontsize=11)
plt.savefig("Regresion_grad_desc.png")
plt.show()

plt.plot(arrnum_dataX,arrnum_dataY,"ro",label="Datos de entrenamiento")
plt.plot(end,start,label="Regresion + Puntos")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Regresion Lineal')
plt.text(0.6, 0.6, r'$f(x) = (%s) + (%s)*Weight $'%(str(var_intercept)[0:6],str(var_slope)[0:6]), fontsize=11)
plt.savefig("Junto_grad_desc.png")
plt.show()
