# Algoritmo PSO (Particle Swarm Optimization) para el problema TSP (Traveling Salesman Problem)
# Autor: Alejandra Tabares 
# Fecha: 05/04/2025

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

# Definición de la clase Partícula
class Particula:
    def __init__(self, num_ciudades):
        # Inicialización de la posición (una permutación de ciudades)
        self.posicion = list(range(num_ciudades))
        random.shuffle(self.posicion)
        
        # Inicialización de la velocidad (lista de intercambios)
        self.velocidad = []
        
        # Mejor posición encontrada por esta partícula
        self.mejor_posicion = self.posicion.copy()
        
        # Valor de fitness (distancia total del recorrido)
        self.fitness = float('inf')
        self.mejor_fitness = float('inf')
    
    def calcular_fitness(self, matriz_distancias):
        """Calcula la distancia total del recorrido"""
        distancia_total = 0
        for i in range(len(self.posicion)):
            desde = self.posicion[i]
            hasta = self.posicion[(i + 1) % len(self.posicion)]
            distancia_total += matriz_distancias[desde][hasta]
        return distancia_total
    
    def actualizar_mejor_posicion(self):
        """Actualiza la mejor posición si la actual es mejor"""
        if self.fitness < self.mejor_fitness:
            self.mejor_fitness = self.fitness
            self.mejor_posicion = self.posicion.copy()
    
    def aplicar_velocidad(self):
        """Aplica los intercambios de la velocidad a la posición actual"""
        for intercambio in self.velocidad:
            i, j = intercambio
            self.posicion[i], self.posicion[j] = self.posicion[j], self.posicion[i]
    
    def actualizar_velocidad(self, mejor_global, w=0.5, c1=1.5, c2=1.5):
        """
        Actualiza la velocidad de la partícula
        
        Parámetros:
        - mejor_global: Mejor solución encontrada por el enjambre
        - w: Factor de inercia
        - c1: Factor cognitivo
        - c2: Factor social
        
        En PSO para TSP, la velocidad es una lista de operaciones de intercambio (swaps)
        """
        # Componente de inercia: mantener parte de la velocidad anterior
        nueva_velocidad = []
        if random.random() < w and self.velocidad:
            num_swaps = max(1, int(w * len(self.velocidad)))
            nueva_velocidad.extend(random.sample(self.velocidad, num_swaps))
        
        # Componente cognitivo: moverse hacia la mejor posición personal
        if random.random() < c1:
            # Calcular diferencias entre posición actual y mejor personal
            swaps_cognitivos = self.calcular_swaps(self.posicion, self.mejor_posicion)
            num_swaps = max(1, int(c1 * len(swaps_cognitivos)))
            if swaps_cognitivos:
                nueva_velocidad.extend(random.sample(swaps_cognitivos, min(num_swaps, len(swaps_cognitivos))))
        
        # Componente social: moverse hacia la mejor posición global
        if random.random() < c2:
            # Calcular diferencias entre posición actual y mejor global
            swaps_sociales = self.calcular_swaps(self.posicion, mejor_global)
            num_swaps = max(1, int(c2 * len(swaps_sociales)))
            if swaps_sociales:
                nueva_velocidad.extend(random.sample(swaps_sociales, min(num_swaps, len(swaps_sociales))))
        
        self.velocidad = nueva_velocidad
    
    def calcular_swaps(self, ruta1, ruta2):
        """
        Calcula los intercambios necesarios para transformar ruta1 en ruta2
        Implementación del algoritmo de ordenamiento por intercambios
        """
        swaps = []
        ruta_temp = ruta1.copy()
        
        for i in range(len(ruta1)):
            if ruta_temp[i] != ruta2[i]:
                j = ruta_temp.index(ruta2[i])
                ruta_temp[i], ruta_temp[j] = ruta_temp[j], ruta_temp[i]
                swaps.append((i, j))
        
        return swaps

# Clase principal del algoritmo PSO para TSP
class PSO_TSP:
    def __init__(self, ciudades, tam_poblacion=50, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        """
        Inicializa el algoritmo PSO para TSP
        
        Parámetros:
        - ciudades: Lista de coordenadas (x, y) de las ciudades
        - tam_poblacion: Número de partículas en el enjambre
        - max_iter: Número máximo de iteraciones
        - w: Factor de inercia
        - c1: Factor cognitivo
        - c2: Factor social
        """
        self.ciudades = ciudades
        self.num_ciudades = len(ciudades)
        self.tam_poblacion = tam_poblacion
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Calcular matriz de distancias entre ciudades
        self.matriz_distancias = self.calcular_matriz_distancias()
        
        # Inicializar población de partículas
        self.particulas = [Particula(self.num_ciudades) for _ in range(tam_poblacion)]
        
        # Evaluar fitness inicial de cada partícula
        for p in self.particulas:
            p.fitness = p.calcular_fitness(self.matriz_distancias)
            p.mejor_fitness = p.fitness
        
        # Inicializar mejor solución global
        self.mejor_global_posicion = min(self.particulas, key=lambda p: p.fitness).posicion.copy()
        self.mejor_global_fitness = min(self.particulas, key=lambda p: p.fitness).fitness
        
        # Para almacenar el historial de mejores soluciones
        self.historial_fitness = []
        self.historial_rutas = []
    
    def calcular_matriz_distancias(self):
        """Calcula la matriz de distancias euclidianas entre todas las ciudades"""
        matriz = np.zeros((self.num_ciudades, self.num_ciudades))
        for i in range(self.num_ciudades):
            for j in range(self.num_ciudades):
                if i != j:
                    x1, y1 = self.ciudades[i]
                    x2, y2 = self.ciudades[j]
                    matriz[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return matriz
    
    def ejecutar(self, animar=True):
        """Ejecuta el algoritmo PSO"""
        print("Iniciando PSO para TSP con {} ciudades y {} partículas...".format(
            self.num_ciudades, self.tam_poblacion))
        
        # Para animación
        if animar:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Configurar gráfico de ciudades
            x = [ciudad[0] for ciudad in self.ciudades]
            y = [ciudad[1] for ciudad in self.ciudades]
            ax1.scatter(x, y, c='red', s=50)
            for i, (x, y) in enumerate(self.ciudades):
                ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            ax1.set_title('Mejor ruta encontrada')
            
            # Configurar gráfico de convergencia
            ax2.set_title('Convergencia del algoritmo')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Distancia total')
            
            line1, = ax1.plot([], [], 'b-', lw=2)
            line2, = ax2.plot([], [], 'g-', lw=2)
            
            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                return line1, line2
            
            def update(frame):
                # Ejecutar una iteración del algoritmo
                self.iterar()
                
                # Actualizar gráfico de ruta
                ruta = self.mejor_global_posicion
                x_ruta = [self.ciudades[ruta[i]][0] for i in range(self.num_ciudades)]
                y_ruta = [self.ciudades[ruta[i]][1] for i in range(self.num_ciudades)]
                # Cerrar el ciclo
                x_ruta.append(self.ciudades[ruta[0]][0])
                y_ruta.append(self.ciudades[ruta[0]][1])
                line1.set_data(x_ruta, y_ruta)
                
                # Actualizar gráfico de convergencia
                line2.set_data(range(len(self.historial_fitness)), self.historial_fitness)
                ax2.relim()
                ax2.autoscale_view()
                
                # Mostrar información de la iteración actual
                print(f"Iteración {frame+1}: Mejor distancia = {self.mejor_global_fitness:.2f}")
                
                return line1, line2
            
            ani = FuncAnimation(fig, update, frames=range(self.max_iter),
                                init_func=init, blit=True, repeat=False, interval=200)
            plt.tight_layout()
            plt.show()
            
        else:
            # Ejecución sin animación
            for i in range(self.max_iter):
                self.iterar()
                print(f"Iteración {i+1}: Mejor distancia = {self.mejor_global_fitness:.2f}")
            
            # Mostrar resultado final
            self.mostrar_resultado()
    
    def iterar(self):
        """Realiza una iteración del algoritmo PSO"""
        for p in self.particulas:
            # Aplicar velocidad actual para obtener nueva posición
            p.aplicar_velocidad()
            
            # Evaluar nueva posición
            p.fitness = p.calcular_fitness(self.matriz_distancias)
            
            # Actualizar mejor posición personal
            p.actualizar_mejor_posicion()
            
            # Actualizar mejor posición global si es necesario
            if p.fitness < self.mejor_global_fitness:
                self.mejor_global_fitness = p.fitness
                self.mejor_global_posicion = p.posicion.copy()
        
        # Actualizar velocidades de todas las partículas
        for p in self.particulas:
            p.actualizar_velocidad(self.mejor_global_posicion, self.w, self.c1, self.c2)
        
        # Guardar historial para análisis
        self.historial_fitness.append(self.mejor_global_fitness)
        self.historial_rutas.append(self.mejor_global_posicion.copy())
    
    def mostrar_resultado(self):
        """Muestra el resultado final del algoritmo"""
        print("\nResultado final:")
        print(f"Mejor distancia encontrada: {self.mejor_global_fitness:.2f}")
        print(f"Mejor ruta: {self.mejor_global_posicion}")
        
        # Graficar la mejor ruta
        plt.figure(figsize=(10, 8))
        
        # Graficar ciudades
        x = [ciudad[0] for ciudad in self.ciudades]
        y = [ciudad[1] for ciudad in self.ciudades]
        plt.scatter(x, y, c='red', s=50)
        
        # Etiquetar ciudades
        for i, (x, y) in enumerate(self.ciudades):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Graficar ruta
        ruta = self.mejor_global_posicion
        for i in range(self.num_ciudades):
            ciudad_actual = ruta[i]
            ciudad_siguiente = ruta[(i + 1) % self.num_ciudades]
            plt.plot([self.ciudades[ciudad_actual][0], self.ciudades[ciudad_siguiente][0]],
                     [self.ciudades[ciudad_actual][1], self.ciudades[ciudad_siguiente][1]], 'b-')
        
        plt.title(f'Mejor ruta encontrada (Distancia: {self.mejor_global_fitness:.2f})')
        plt.grid(True)
        plt.show()
        
        # Graficar convergencia
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historial_fitness)), self.historial_fitness, 'g-')
        plt.title('Convergencia del algoritmo PSO')
        plt.xlabel('Iteración')
        plt.ylabel('Distancia total')
        plt.grid(True)
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Generar ciudades aleatorias
    np.random.seed(42)  # Para reproducibilidad
    num_ciudades = 20
    ciudades = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(num_ciudades)]
    
    # Configurar y ejecutar el algoritmo PSO
    pso = PSO_TSP(
        ciudades=ciudades,
        tam_poblacion=50,
        max_iter=100,
        w=0.5,  # Factor de inercia
        c1=1.5,  # Factor cognitivo
        c2=1.5   # Factor social
    )
    
    # Ejecutar el algoritmo con animación
    pso.ejecutar(animar=True)
