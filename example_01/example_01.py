
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

x = []
y = []
sum_x = 0
sum_y = 0
mul_x_y = 0
sqr_x = 0
wi = 0
wo = 0

with open("100.csv",'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))
N = len(x)

print("Jean Pierre Soto Chirinos")
print("Mínimos cuadrados \n")

print("Datos Utilizados:")



plt.plot(x,y,"ro",label="Datos de entrenamiento")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Puntos de entrenamiento')
plt.savefig("Puntos_min_sqr.png")
plt.show()

for i in range(0, len(x)):
    print(str(x[i])+ "       "+str(y[i]))
    sum_x += x[i]
    sum_y += y[i]
    mul_x_y += y[i]*x[i]
    sqr_x += x[i]**2
wi=((N*mul_x_y)-(sum_x*sum_y))/((N*sqr_x)-(sum_x**2))
wo=(sum_y-(wi*sum_x))/N
print("\nSuma de x: ",sum_x)
print("Suma de y: ",sum_y)
print("Suma de x*y: ",mul_x_y)
print("Suma de x^2: ",sqr_x)
print("\nIntercepto: ",wo)
print("Pendiente: ",wi)


start= [wo+wi*y[0],wo+wi*y[N-1]]
end= [y[0],y[N-1]]
plt.plot(end,start,label="Regresion")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Regresion Lineal')
plt.text(0.6, 0.6, r'$f(x) = (%s) + (%s)*Weight $'%(str(wo)[0:6],str(wi)[0:6]), fontsize=11)
plt.savefig("Regresion_min_sqr.png")
plt.show()

plt.plot(x,y,"ro",label="Datos de entrenamiento")
plt.plot(end,start,label="Regresion + Puntos")
plt.xlabel('Var X')
plt.ylabel('Var Y')
plt.title('Regresion Lineal')
plt.text(0.6, 0.6, r'$f(x) = (%s) + (%s)*Weight $'%(str(wo)[0:6],str(wi)[0:6]), fontsize=11)
plt.savefig("Junto_min_sqr.png")
plt.show()
