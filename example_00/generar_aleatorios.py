import numpy as np
import math

# Definiendo constantes
N = 8 # numero de puntos
FI = 2/(pow(5,1/2)-1) # constante FI
PI = math.pi # constante PI

def main():
    #Arreglo con numeros
    f= open(str(100)+".csv","w+")
    arrnum_dataX = np.linspace(-100, 110, 100)   
    arrnum_dataY = arrnum_dataX + np.random.randn(*arrnum_dataX.shape) * FI + PI

    for i in range(0, len(arrnum_dataX)):
            
        f.write("%d,%d \n" % (arrnum_dataX[i] , arrnum_dataY[i]))
        #cerramos el archivo
    f.close()

if __name__ == '__main__':
   main()