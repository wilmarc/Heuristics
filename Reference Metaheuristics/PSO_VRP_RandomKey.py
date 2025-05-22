import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math
from collections import defaultdict

class Particula_VRP:
    """Representa una partícula en el algoritmo PSO para VRP con Random Keys"""
    
    def __init__(self, num_clientes, num_vehiculos):
        self.num_clientes = num_clientes
        self.num_vehiculos = num_vehiculos
        # Inicializar random keys: valores entre 0 y 1 para cada cliente
        self.posicion = np.random.random(num_clientes)
        # Inicializar velocidad aleatoria entre -0.1 y 0.1
        self.velocidad = np.random.uniform(-0.1, 0.1, num_clientes)
        # Mejor posición personal
        self.mejor_posicion = self.posicion.copy()
        # Fitness (costo de la ruta)
        self.fitness = float('inf')
        self.mejor_fitness = float('inf')
        # Rutas decodificadas
        self.rutas = []
    
    def actualizar_velocidad(self, mejor_global, w, c1, c2):
        """Actualiza la velocidad de la partícula"""
        r1 = np.random.random(self.num_clientes)
        r2 = np.random.random(self.num_clientes)
        
        # Componente de inercia
        inercia = w * self.velocidad
        
        # Componente cognitiva (atracción hacia la mejor posición personal)
        cognitiva = c1 * r1 * (self.mejor_posicion - self.posicion)
        
        # Componente social (atracción hacia la mejor posición global)
        social = c2 * r2 * (mejor_global - self.posicion)
        
        # Actualizar velocidad
        self.velocidad = inercia + cognitiva + social
        
        # Limitar velocidad
        self.velocidad = np.clip(self.velocidad, -0.1, 0.1)
    
    def actualizar_posicion(self):
        """Actualiza la posición de la partícula"""
        self.posicion += self.velocidad
        
        # Asegurar que los valores estén entre 0 y 1
        self.posicion = np.clip(self.posicion, 0, 1)
    
    def decodificar_rutas(self, clientes, deposito, capacidad_vehiculo):
        """Decodifica las random keys en rutas de vehículos"""
        # Crear pares (índice_cliente, random_key)
        pares = [(i, self.posicion[i]) for i in range(self.num_clientes)]
        
        # Ordenar clientes por su valor de random key
        pares_ordenados = sorted(pares, key=lambda x: x[1])
        
        # Inicializar rutas
        rutas = [[] for _ in range(self.num_vehiculos)]
        carga_actual = [0] * self.num_vehiculos
        
        # Asignar clientes a vehículos
        for cliente_idx, _ in pares_ordenados:
            # Calcular a qué vehículo asignar este cliente
            # Usamos el valor de random key para determinar el vehículo
            vehiculo_idx = 0
            min_carga = float('inf')
            
            # Asignar al vehículo con menor carga que pueda llevar la demanda
            for v in range(self.num_vehiculos):
                if (carga_actual[v] + clientes[cliente_idx][2] <= capacidad_vehiculo and 
                    carga_actual[v] < min_carga):
                    min_carga = carga_actual[v]
                    vehiculo_idx = v
            
            # Si no se puede asignar a ningún vehículo, crear uno nuevo si es posible
            if min_carga == float('inf'):
                # Buscar el vehículo con menor carga
                vehiculo_idx = carga_actual.index(min(carga_actual))
            
            # Asignar cliente al vehículo
            rutas[vehiculo_idx].append(cliente_idx)
            carga_actual[vehiculo_idx] += clientes[cliente_idx][2]
        
        self.rutas = rutas
        return rutas
    
    def calcular_fitness(self, clientes, deposito, capacidad_vehiculo):
        """Calcula el fitness (distancia total) de la partícula"""
        rutas = self.decodificar_rutas(clientes, deposito, capacidad_vehiculo)
        distancia_total = 0
        
        for ruta in rutas:
            if not ruta:  # Si la ruta está vacía
                continue
                
            # Distancia desde el depósito al primer cliente
            distancia_total += calcular_distancia(deposito, clientes[ruta[0]])
            
            # Distancia entre clientes consecutivos
            for i in range(len(ruta) - 1):
                distancia_total += calcular_distancia(clientes[ruta[i]], clientes[ruta[i + 1]])
            
            # Distancia desde el último cliente al depósito
            distancia_total += calcular_distancia(clientes[ruta[-1]], deposito)
        
        # Penalización por exceder la capacidad
        penalizacion = 0
        for i, ruta in enumerate(rutas):
            carga = sum(clientes[cliente_idx][2] for cliente_idx in ruta)
            if carga > capacidad_vehiculo:
                penalizacion += (carga - capacidad_vehiculo) * 100
        
        self.fitness = distancia_total + penalizacion
        
        # Actualizar mejor posición personal si es necesario
        if self.fitness < self.mejor_fitness:
            self.mejor_fitness = self.fitness
            self.mejor_posicion = self.posicion.copy()
        
        return self.fitness

def calcular_distancia(punto1, punto2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)

class PSO_VRP:
    """Implementación del algoritmo PSO para el problema VRP usando Random Keys"""
    
    def __init__(self, clientes, deposito, capacidad_vehiculo, num_vehiculos, 
                 tam_poblacion=50, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        """
        Inicializa el algoritmo PSO para VRP
        
        Args:
            clientes: Lista de tuplas (x, y, demanda) para cada cliente
            deposito: Tupla (x, y) con la ubicación del depósito
            capacidad_vehiculo: Capacidad máxima de cada vehículo
            num_vehiculos: Número de vehículos disponibles
            tam_poblacion: Tamaño de la población de partículas
            max_iter: Número máximo de iteraciones
            w: Factor de inercia
            c1: Factor cognitivo
            c2: Factor social
        """
        self.clientes = clientes
        self.deposito = deposito
        self.capacidad_vehiculo = capacidad_vehiculo
        self.num_vehiculos = num_vehiculos
        self.num_clientes = len(clientes)
        self.tam_poblacion = tam_poblacion
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Inicializar partículas
        self.particulas = [Particula_VRP(self.num_clientes, self.num_vehiculos) 
                          for _ in range(tam_poblacion)]
        
        # Evaluar fitness inicial
        for p in self.particulas:
            p.calcular_fitness(self.clientes, self.deposito, self.capacidad_vehiculo)
        
        # Inicializar mejor global
        self.mejor_global_fitness = float('inf')
        self.mejor_global_posicion = np.zeros(self.num_clientes)
        self.mejor_global_rutas = []
        
        # Para análisis y visualización
        self.historial_fitness = []
        self.historial_rutas = []
    
    def ejecutar(self, animar=False):
        """Ejecuta el algoritmo PSO"""
        # Encontrar mejor global inicial
        for p in self.particulas:
            if p.fitness < self.mejor_global_fitness:
                self.mejor_global_fitness = p.fitness
                self.mejor_global_posicion = p.posicion.copy()
                self.mejor_global_rutas = p.rutas
        
        # Guardar estado inicial
        self.historial_fitness.append(self.mejor_global_fitness)
        self.historial_rutas.append(self.mejor_global_rutas)
        
        # Configurar animación si es necesario
        if animar:
            fig, ax = plt.subplots(figsize=(10, 8))
            anim = FuncAnimation(fig, self.actualizar_animacion, frames=self.max_iter,
                                 fargs=(ax,), interval=200, repeat=False)
            plt.show()
        else:
            # Ejecutar sin animación
            for i in range(self.max_iter):
                self.iterar()
                if (i + 1) % 10 == 0:
                    print(f"Iteración {i+1}/{self.max_iter}, Mejor fitness: {self.mejor_global_fitness:.2f}")
            
            # Mostrar resultado final
            self.mostrar_resultado()
    
    def iterar(self):
        """Realiza una iteración del algoritmo PSO"""
        # Actualizar posiciones
        for p in self.particulas:
            p.actualizar_posicion()
            p.calcular_fitness(self.clientes, self.deposito, self.capacidad_vehiculo)
        
        # Actualizar mejor global
        for p in self.particulas:
            if p.fitness < self.mejor_global_fitness:
                self.mejor_global_fitness = p.fitness
                self.mejor_global_posicion = p.posicion.copy()
                self.mejor_global_rutas = p.rutas
        
        # Actualizar velocidades
        for p in self.particulas:
            p.actualizar_velocidad(self.mejor_global_posicion, self.w, self.c1, self.c2)
        
        # Guardar historial
        self.historial_fitness.append(self.mejor_global_fitness)
        self.historial_rutas.append(self.mejor_global_rutas)
    
    def actualizar_animacion(self, frame, ax):
        """Actualiza la animación en cada frame"""
        self.iterar()
        
        # Limpiar el gráfico
        ax.clear()
        
        # Graficar depósito
        ax.scatter(self.deposito[0], self.deposito[1], c='red', s=100, marker='s')
        ax.annotate("Depósito", (self.deposito[0], self.deposito[1]), 
                   xytext=(5, 5), textcoords='offset points')
        
        # Graficar clientes
        for i, (x, y, demanda) in enumerate(self.clientes):
            ax.scatter(x, y, c='blue', s=50)
            ax.annotate(f"{i}({demanda})", (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Graficar rutas
        colores = plt.cm.tab10(np.linspace(0, 1, self.num_vehiculos))
        
        for i, ruta in enumerate(self.mejor_global_rutas):
            if not ruta:  # Si la ruta está vacía
                continue
                
            color = colores[i % len(colores)]
            
            # Línea desde el depósito al primer cliente
            ax.plot([self.deposito[0], self.clientes[ruta[0]][0]],
                   [self.deposito[1], self.clientes[ruta[0]][1]], 
                   c=color, linestyle='-', linewidth=2)
            
            # Líneas entre clientes
            for j in range(len(ruta) - 1):
                ax.plot([self.clientes[ruta[j]][0], self.clientes[ruta[j+1]][0]],
                       [self.clientes[ruta[j]][1], self.clientes[ruta[j+1]][1]],
                       c=color, linestyle='-', linewidth=2)
            
            # Línea desde el último cliente al depósito
            ax.plot([self.clientes[ruta[-1]][0], self.deposito[0]],
                   [self.clientes[ruta[-1]][1], self.deposito[1]],
                   c=color, linestyle='-', linewidth=2)
        
        # Información de la iteración
        ax.set_title(f'Iteración {frame+1}, Mejor distancia: {self.mejor_global_fitness:.2f}')
        ax.grid(True)
        
        return ax
    
    def mostrar_resultado(self):
        """Muestra el resultado final del algoritmo"""
        print("\nResultado final:")
        print(f"Mejor distancia encontrada: {self.mejor_global_fitness:.2f}")
        
        # Mostrar rutas
        for i, ruta in enumerate(self.mejor_global_rutas):
            if ruta:  # Si la ruta no está vacía
                carga = sum(self.clientes[cliente_idx][2] for cliente_idx in ruta)
                print(f"Ruta {i+1}: {[r for r in ruta]} - Carga: {carga}/{self.capacidad_vehiculo}")
        
        # Graficar la mejor solución
        plt.figure(figsize=(10, 8))
        
        # Graficar depósito
        plt.scatter(self.deposito[0], self.deposito[1], c='red', s=100, marker='s')
        plt.annotate("Depósito", (self.deposito[0], self.deposito[1]), 
                    xytext=(5, 5), textcoords='offset points')
        
        # Graficar clientes
        for i, (x, y, demanda) in enumerate(self.clientes):
            plt.scatter(x, y, c='blue', s=50)
            plt.annotate(f"{i}({demanda})", (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Graficar rutas
        colores = plt.cm.tab10(np.linspace(0, 1, self.num_vehiculos))
        
        for i, ruta in enumerate(self.mejor_global_rutas):
            if not ruta:  # Si la ruta está vacía
                continue
                
            color = colores[i % len(colores)]
            
            # Línea desde el depósito al primer cliente
            plt.plot([self.deposito[0], self.clientes[ruta[0]][0]],
                    [self.deposito[1], self.clientes[ruta[0]][1]], 
                    c=color, linestyle='-', linewidth=2, label=f"Vehículo {i+1}")
            
            # Líneas entre clientes
            for j in range(len(ruta) - 1):
                plt.plot([self.clientes[ruta[j]][0], self.clientes[ruta[j+1]][0]],
                        [self.clientes[ruta[j]][1], self.clientes[ruta[j+1]][1]],
                        c=color, linestyle='-', linewidth=2)
            
            # Línea desde el último cliente al depósito
            plt.plot([self.clientes[ruta[-1]][0], self.deposito[0]],
                    [self.clientes[ruta[-1]][1], self.deposito[1]],
                    c=color, linestyle='-', linewidth=2)
        
        plt.title(f'Mejor solución encontrada (Distancia: {self.mejor_global_fitness:.2f})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Graficar convergencia
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historial_fitness)), self.historial_fitness, 'g-')
        plt.title('Convergencia del algoritmo PSO para VRP')
        plt.xlabel('Iteración')
        plt.ylabel('Distancia total')
        plt.grid(True)
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración del problema
    np.random.seed(42)  # Para reproducibilidad
    
    # Depósito en el centro
    deposito = (50, 50)
    
    # Generar clientes aleatorios con demandas
    num_clientes = 20
    clientes = []
    for _ in range(num_clientes):
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 100)
        demanda = np.random.randint(1, 10)
        clientes.append((x, y, demanda))
    
    # Configuración del VRP
    capacidad_vehiculo = 30
    num_vehiculos = 5
    
    # Configurar y ejecutar el algoritmo PSO
    pso = PSO_VRP(
        clientes=clientes,
        deposito=deposito,
        capacidad_vehiculo=capacidad_vehiculo,
        num_vehiculos=num_vehiculos,
        tam_poblacion=50,
        max_iter=400,
        w=0.5,  # Factor de inercia
        c1=1.5,  # Factor cognitivo
        c2=1.5   # Factor social
    )
    
    # Ejecutar el algoritmo con animación
    pso.ejecutar(animar=True)
