import math
import csv 
import numpy as np
from tqdm import tqdm
#def booth function (x+2y-7)^2 + (2x + y -5)^2

def booth_function(fitness):
    return (math.pow((fitness[0]+(2*fitness[1])-7),2)+math.pow(((2*fitness[0])+fitness[1]-5),2))

print("Jean Pierre Soto Chirinos")
print("Algorimos Geneticos")
print("Caso de estudio : Funcion Booth\n")
#Inicialización de variables

num_population = 20
num_dimensions = 2

num_part_per_tournament = 2

const_cross_prob = .7
const_blx_alpha = .5
const_mutation_prob = .05

const_steps = 5000

#Inicialización de poblacion
np.random.seed(5043)
arr_population = np.random.uniform(-10,10,(num_population,num_dimensions))

#Imprimir parametros
print("Parametros: ")
print("Numero de población: ",num_population)
print("Numero de dimensión por individuo: ",num_dimensions)
print("Tupla Matin Pool: ",num_part_per_tournament)
print("Prob. de Cruzamiento: ",const_cross_prob)
print("BLX Alpha : ",const_blx_alpha)
print("Prob. de mutación: ",const_mutation_prob)
print("Mutación Uniforme ")
print("Cantidad de iteraciones: ",const_steps,"\n")

# loop de iteraciones
for step in tqdm(range(0, const_steps)):

    #inicialización de arr auxiliares
    arr_fitness = []
    arr_parents = []

    arr_num_parents = []
    arr_new_population = []

    arr_matin_pool = []

    #genera tabla del matin pool
    a =  np.arange(num_population)
    arr_matin_pool = np.random.choice(a,(num_population,num_dimensions))

    print("Población", step)
    #se calcula los fitness por indivuduo
    for i in range(0,len(arr_population)):
        print(i+1,") ",arr_population[i])
    print("\n")
    print("Calcular aptitud por individuo")
    for i in range(0,len(arr_population)):
        print(i+1,") ",arr_population[i],"  ",booth_function(arr_population[i]))
        arr_fitness.append(booth_function(arr_population[i]))

    for best in arr_matin_pool:
        
        #descartar repetidos en el array de matinpool generado
        while(best[0]==best[1]):
            best[0]=np.random.randint(0,16)

        if(arr_fitness[best[0]]<arr_fitness[best[1]]):
            arr_parents.append(arr_population[best[0]])
            arr_num_parents.append(best[0])
        else:
            arr_parents.append(arr_population[best[1]])
            arr_num_parents.append(best[1])
    print("\n")
    for i in range(0,len(arr_matin_pool)):
        print(arr_matin_pool[i][0]+1,"  vs  ",arr_matin_pool[i][1]+1,"  =>  ",arr_num_parents[i],"  =>  ",arr_population[arr_num_parents[i]])

    print("\n")
    #general la tabla de tuplas de padres
    arr_parent_tuple=[]
    arr_parent_tuple = np.random.choice(a,(num_population,num_dimensions))

    for best in arr_parent_tuple:
        #incialización de variables de probabilidades con numpy
        mutation = np.random.choice([True,False], 1, p=[0.05,0.95])[0]
        cross = np.random.choice([True,False], 1, p=[0.7,0.3])[0]
        index_mutation = np.random.choice([True,False], 1, p=[0.7,0.3])[0]

        #descartar repetidos generados por las tuplas de cruzamiento de padres
        while(best[0]==best[1]):
            best[0]=np.random.randint(0,16)
        print("Selección de padres")
        print(best[0]+1," - ",best[1]+1,"->",arr_num_parents[best[0]]," - ",arr_num_parents[best[1]]," => ",arr_parents[best[0]]," - ",arr_parents[best[1]])
        new_x = 0
        new_y = 0
        new_point = []
        blx_alpha_0 = 0
        blx_alpha_1 = 0 
        no_cross_index = 0
        mutation_index = 0

        if(cross):
            print("Cruzamiento")
            blx_alpha_0=np.random.uniform(-const_blx_alpha,1+const_blx_alpha)
            blx_alpha_1=np.random.uniform(-const_blx_alpha,1+const_blx_alpha)
            print("Beta 1: ",blx_alpha_0)
            print("Beta 2: ",blx_alpha_1)
            new_x = arr_parents[best[0]][0]+(blx_alpha_0*(arr_parents[best[1]][0]-arr_parents[best[0]][0]))
            new_y = arr_parents[best[0]][1]+(blx_alpha_1*(arr_parents[best[1]][1]-arr_parents[best[0]][1]))
        else:
            print("Sin cruzamiento")
            no_cross_index = np.random.choice([0,1], 1, p=[0.5,0.5])[0]
            new_x = arr_parents[best[no_cross_index]][0]
            new_y = arr_parents[best[no_cross_index]][1]
        
        new_point = [new_x,new_y]
        print(new_point)
        
        if(mutation):
            print("Mutación")
            mutation_index = np.random.choice([0,1], 1, p=[0.5,0.5])[0]
            new_point[mutation_index] = np.random.uniform(-10,10)
            print(new_point)
        else:
            print("Sin Mutacion")

        arr_new_population.append(new_point)
        print("\n")

    arr_population = arr_new_population
  

print("Población final")
#se calcula los fitness por indivuduo
for i in range(0,len(arr_population)):
    print(i+1,") ",arr_population[i])
print("\n")
print("Calcular aptitud por individuo")
for i in range(0,len(arr_population)):
    print(i+1,") ",arr_population[i],"  ",booth_function(arr_population[i]))
    arr_fitness.append(booth_function(arr_population[i]))