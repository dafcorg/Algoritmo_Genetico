import numpy as np
import random
import matplotlib.pyplot as plt

# Parámetros del problema
NUM_EMPLEADOS = 12
DIAS_SEMANA = 7
DIAS_TRABAJO = 5
POBLACION_INICIAL = 50
GENERACIONES = 100

# Generación de una matriz binaria de turnos inicial
def generar_individuo():
    individuo = []
    for _ in range(NUM_EMPLEADOS):
        dias_trabajo = random.sample(range(DIAS_SEMANA), DIAS_TRABAJO)
        empleado = [1 if dia in dias_trabajo else 0 for dia in range(DIAS_SEMANA)]
        individuo.append(empleado)
    return np.array(individuo)

# Función de fitness
def calcular_fitness(individuo):
    fitness = 0
    penalizacion = 0
    for dia in range(DIAS_SEMANA):
        cobertura_dia = sum(individuo[:, dia])
        fitness += cobertura_dia
        if cobertura_dia == 0:  # Penalización si algún día está sin cobertura
            penalizacion += 50
    for empleado in individuo:
        if sum(empleado) != DIAS_TRABAJO:
            penalizacion += 10
    return fitness - penalizacion

# Selección, cruce y mutación
def seleccion(poblacion, fitness_poblacion):
    padres = random.choices(poblacion, weights=fitness_poblacion, k=2)
    return padres

def cruce(padre1, padre2):
    punto_cruce = random.randint(1, DIAS_SEMANA - 1)
    hijo1 = np.vstack((padre1[:punto_cruce, :], padre2[punto_cruce:, :]))
    hijo2 = np.vstack((padre2[:punto_cruce, :], padre1[punto_cruce:, :]))
    return hijo1, hijo2

def mutacion(individuo):
    for i in range(NUM_EMPLEADOS):
        if random.random() < 0.1:  # 10% de probabilidad de mutación
            dias_trabajo = np.where(individuo[i] == 1)[0]
            dias_descanso = np.where(individuo[i] == 0)[0]
            if len(dias_trabajo) > 0 and len(dias_descanso) > 0:
                dia_trabajo = random.choice(dias_trabajo)
                dia_descanso = random.choice(dias_descanso)
                individuo[i][dia_trabajo], individuo[i][dia_descanso] = 0, 1
    return individuo

# Algoritmo genético principal
poblacion = [generar_individuo() for _ in range(POBLACION_INICIAL)]

for generacion in range(GENERACIONES):
    fitness_poblacion = [calcular_fitness(ind) for ind in poblacion]
    nueva_poblacion = []
    for _ in range(POBLACION_INICIAL // 2):
        padre1, padre2 = seleccion(poblacion, fitness_poblacion)
        hijo1, hijo2 = cruce(padre1, padre2)
        nueva_poblacion.extend([mutacion(hijo1), mutacion(hijo2)])
    poblacion = nueva_poblacion

# Obtener la mejor solución de la última generación
mejor_solucion = max(poblacion, key=calcular_fitness)
fitness_mejor_solucion = calcular_fitness(mejor_solucion)

# Función para graficar la matriz de turnos
def graficar_turnos(matriz_turnos, fitness):
    plt.figure(figsize=(10, 6))
    plt.imshow(matriz_turnos, cmap='Blues', aspect='auto')
    plt.colorbar(label="Turno (1 = Trabaja, 0 = Descansa)")
    plt.title(f"Distribución de Turnos Semanales - Fitness: {fitness}")
    plt.xlabel("Días de la Semana (Lunes a Domingo)")
    plt.ylabel("Empleados (1 a 12)")
    plt.xticks(ticks=range(DIAS_SEMANA), labels=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
    plt.yticks(ticks=range(NUM_EMPLEADOS), labels=[f"Empleado {i+1}" for i in range(NUM_EMPLEADOS)])
    plt.show()

# Graficar la mejor solución
graficar_turnos(mejor_solucion, fitness_mejor_solucion)
