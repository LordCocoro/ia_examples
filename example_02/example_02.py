import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import numpy as np

x = []
y = []
o_y=[]

const_steps = 1000000
const_learningRate = 0.0001
const_umbralCriteria = 0.8

var_slope = 0.904561
var_intercept = -2.61146
s =0
def sigmoid(x):
 
    y=1/(1+np.exp((var_slope*x+var_intercept)*-1))
    return y
 
 
def plot_sigmoid(a,b,s):
    # param: punto de inicio, punto final, espaciado
    x = np.arange(0, 25, 0.2)
    y = sigmoid(x)
    plt.plot(a,b,"ro",label="Datos de entrenamiento")
    plt.plot(x, y)
    plt.savefig(str(s)+".png")
    plt.show()

with open("100.csv",'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))

print("Jean Pierre Soto Chirinos")
print("Regresión logística \n")
print("Pendiente anterior: ",var_slope)
print("Intercepto anterior: ",var_intercept)
print("Taza de aprendizaje: "),const_learningRate
print("Umbral: ",const_umbralCriteria)
print("Datos de entrenamiento:")
print("x: ",x)
print("y: ",y)
print("Datos de Test")
print("x: {3,6}")
print("x: {0,1}")

N = len(x)

plot_sigmoid(x,y,0)


for step in tqdm(range(0, const_steps)):
    s = step
    var_intercept_gradient = 0
    var_slope_gradient = 0
    sum_error = 0
    for i in range(0, len(x)):
        exp_y = 0
        exp_y = (1/(1+math.exp((var_slope*int(x[i])+var_intercept)*-1)))
        sum_error += abs((y[i]*np.log(exp_y))+((1-y[i])*(np.log(1-exp_y))))
        var_intercept_gradient += (exp_y-y[i])
        var_slope_gradient += (exp_y-y[i])*x[i]
    
    if(step%10==0):
        print("\nIteración ", step+1)
        print("Intercepto anterior: ",str(var_intercept)[0:8]," Pendiente anterior: ",str(var_slope)[0:8])    
        print("Error: ",sum_error)
        print("Coeficiente de aprendizaje: ", const_learningRate)
        print("Intercepto gradiente (derivada): ",str(var_intercept_gradient)[0:8]," Pendiente gradiente (derivada): ",str(var_slope_gradient)[0:8])
        print("Intercepto nuevo: ",str(var_intercept - (const_learningRate * var_intercept_gradient))[0:8]," Pendiente nuevo: ",str(var_slope - (const_learningRate * var_slope_gradient))[0:8])
        print("Criterio Intercepto (nuevo) ",str(const_learningRate * var_intercept_gradient)[0:8])
        print("Criterio Pendiente (nuevo)",str(const_learningRate * var_slope_gradient)[0:8])
    var_intercept =  var_intercept - (const_learningRate * var_intercept_gradient)
    var_slope = var_slope - (const_learningRate * var_slope_gradient)
    if(sum_error<=const_umbralCriteria):
         break
rpt_6="Incorrecto"
rpt_3="Incorrecto"
rpt = "0% de acierto"
aciert=0
test_6=1/(1+math.exp(-(var_slope*6+var_intercept)))
test_3=1/(1+math.exp(-(var_slope*3+var_intercept)))
if(round(test_6)>=1):
    rpt_6="Correcto"
    aciert+=1
if(round(test_3)<1):
    rpt_3="Correcto"
    aciert+=1

if(aciert==2):
    rpt = "100% de acierto"
elif(aciert == 1):
    rpt = "50% de acierto"

print("\ntest")
print("test con 6 :",str(test_6)[0:8],"aprovado estimado",round(test_6),", "+rpt_6)
print("test con 3 :",test_3,"aprovado estimado",round(test_3),", "+rpt_3)
print(rpt)
plot_sigmoid(x,y,s)

