import numpy as np
import random
import matplotlib.pyplot as plt

def crear_tablero_inicial(n):
    """
    Crea una solución inicial aleatoria para el problema de las n reinas.
    
    Args:
        n (int): Tamaño del tablero (n x n)
    
    Returns:
        numpy.array: Vector de posiciones donde el índice representa la columna 
                    y el valor representa la fila donde está ubicada la reina
    """
    return np.random.permutation(n)

def calcular_conflictos(estado):
    """
    Calcula el número de pares de reinas que se atacan entre sí.
    
    Args:
        estado (numpy.array): Vector que representa las posiciones de las reinas
    
    Returns:
        int: Número de conflictos (pares de reinas que se atacan)
    """
    n = len(estado)
    conflictos = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Verifica ataques en diagonal
            if abs(estado[i] - estado[j]) == abs(i - j):
                conflictos += 1
                
    return conflictos

def generar_vecino(estado):
    """
    Genera una solución vecina intercambiando la posición de dos reinas al azar. SWAP
    
    Args:
        estado (numpy.array): Estado actual del tablero
    
    Returns:
        numpy.array: Nuevo estado con dos reinas intercambiadas
    """
    nuevo_estado = estado.copy()
    n = len(estado)
    i, j = np.random.randint(0, n, size=2)
    nuevo_estado[i], nuevo_estado[j] = nuevo_estado[j], nuevo_estado[i]
    return nuevo_estado

def visualizar_tablero(estado, titulo=""):
    """
    Visualiza el tablero de ajedrez con las reinas.
    
    Args:
        estado (numpy.array): Vector que representa las posiciones de las reinas
        titulo (str): Título para la visualización
    """
    n = len(estado)
    tablero = np.zeros((n, n))
    
    # Coloca las reinas en el tablero
    for col, fila in enumerate(estado):
        tablero[fila][col] = 1
        
    plt.figure(figsize=(8, 8))
    plt.imshow(tablero, cmap='binary')
    plt.grid(True)
    plt.title(titulo)
    plt.savefig("tablero.png")
    plt.show()

def recocido_simulado_n_reinas(n, T0, Tmin, alpha, max_iter):
    """
    Implementa el algoritmo de recocido simulado para resolver el problema de las n reinas.
    
    Args:
        n (int): Tamaño del tablero (n x n)
        T0 (float): Temperatura inicial
        Tmin (float): Temperatura mínima
        alpha (float): Factor de enfriamiento
        max_iter (int): Número máximo de iteraciones
    
    Returns:
        tuple: (mejor_estado, mejor_costo, historial_costos)
    """
    # Inicialización
    estado_actual = crear_tablero_inicial(n)
    costo_actual = calcular_conflictos(estado_actual)
    mejor_estado = estado_actual.copy()
    mejor_costo = costo_actual
    
    T = T0
    historial_costos = [costo_actual]
    
    # Bucle principal del recocido simulado
    iter_sin_mejora = 0
    i=0
    i_max=5000
    while T > Tmin and iter_sin_mejora < max_iter and  i <i_max:
        i+=1
        if i %100 == 0 and i >0:
            print(f"Iteración {i} de {i_max}")
        # Generar solución vecina
        estado_vecino = generar_vecino(estado_actual)
        costo_vecino = calcular_conflictos(estado_vecino)
        
        # Calcular diferencia de energía
        delta_E = costo_vecino - costo_actual
        
        # Criterio de aceptación de Metropolis
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            estado_actual = estado_vecino
            costo_actual = costo_vecino
            
            # Actualizar mejor solución encontrada
            if costo_actual < mejor_costo:
                mejor_estado = estado_actual.copy()
                mejor_costo = costo_actual
                iter_sin_mejora = 0
            else:
                iter_sin_mejora += 1
        else:
            iter_sin_mejora += 1
            
        # Enfriar temperatura
        T *= alpha
        historial_costos.append(costo_actual)
        
        # Si encontramos una solución perfecta, terminamos
        if mejor_costo == 0:
            break

    return mejor_estado, mejor_costo, historial_costos



if __name__ == "__main__":
    # Parámetros del problema
    n = 10000  # Tamaño del tablero
    random.seed(1)
    np.random.seed(1)    
    # Ejecutar el algoritmo
    T0=200.0
    Tmin=0.01 
    alpha=0.99
    max_iter=500
    mejor_estado, mejor_costo, historial = recocido_simulado_n_reinas(n,T0,Tmin,alpha,max_iter)
    
    # Mostrar resultados
    print(f"\nMejor solución encontrada:")
    print(f"Número de conflictos: {mejor_costo}")
    # print(f"Posiciones de las reinas: {mejor_estado}")
    
    # Visualizar solución
    visualizar_tablero(mejor_estado, f"Solución con {mejor_costo} conflictos")
    
    # Graficar evolución del costo
    plt.figure(figsize=(10, 5))
    plt.plot(historial)
    plt.xlabel('Iteración')
    plt.ylabel('Número de conflictos')
    plt.title('Evolución del número de conflictos. T0: '+str(T0)+' Tmin: '+str(Tmin)+' alpha: '+str(alpha)+' Iters: '+str(max_iter))
    plt.grid(True)
    plt.show()

