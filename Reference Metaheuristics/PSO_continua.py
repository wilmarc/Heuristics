import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
import time

# Definición de la función objetivo (función de Rosenbrock)
def funcion_objetivo(x, y):
    """
    Función de Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Es una función no convexa utilizada como problema de prueba para algoritmos de optimización.
    Tiene un mínimo global en (x,y) = (1,1) donde f(1,1) = 0
    """
    return (1 - x)**2 + 100 * (y - x**2)**2

# Visualización de la función objetivo
def visualizar_funcion():
    """
    Crea una visualización 3D de la función objetivo para entender mejor
    el espacio de búsqueda.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear una malla de puntos
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = funcion_objetivo(X, Y)
    
    # Graficar la superficie
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    
    # Configurar etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.set_title('Función de Rosenbrock')
    
    # Añadir una barra de color
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('funcion_rosenbrock.png')
    return fig, ax

# Implementación del algoritmo PSO (Particle Swarm Optimization)
class PSO:
    """
    Implementación del algoritmo de Optimización por Enjambre de Partículas (PSO).
    
    Parámetros:
    - num_particulas: Número de partículas en el enjambre
    - max_iter: Número máximo de iteraciones
    - limites: Límites del espacio de búsqueda [min_x, max_x, min_y, max_y]
    - w: Factor de inercia
    - c1: Coeficiente cognitivo
    - c2: Coeficiente social
    """
    def __init__(self, num_particulas=30, max_iter=100, limites=[-2, 2, -1, 3], 
                 w=0.7, c1=1.5, c2=1.5):
        self.num_particulas = num_particulas
        self.max_iter = max_iter
        self.limites = limites
        self.w = w  # Factor de inercia
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social
        
        # Inicializar partículas
        self.particulas = []
        self.mejor_global_pos = None
        self.mejor_global_costo = float('inf')
        self.historico_posiciones = []
        
        # Crear partículas iniciales
        for i in range(num_particulas):
            # Posición aleatoria dentro de los límites
            pos_x = random.uniform(limites[0], limites[1])
            pos_y = random.uniform(limites[2], limites[3])
            
            # Velocidad inicial aleatoria
            vel_x = random.uniform(-0.5, 0.5)
            vel_y = random.uniform(-0.5, 0.5)
            
            # Evaluar la posición
            costo = funcion_objetivo(pos_x, pos_y)
            
            # Guardar la partícula
            particula = {
                'posicion': np.array([pos_x, pos_y]),
                'velocidad': np.array([vel_x, vel_y]),
                'costo': costo,
                'mejor_pos': np.array([pos_x, pos_y]),
                'mejor_costo': costo
            }
            self.particulas.append(particula)
            
            # Actualizar el mejor global si es necesario
            if costo < self.mejor_global_costo:
                self.mejor_global_costo = costo
                self.mejor_global_pos = np.array([pos_x, pos_y])
    
    def optimizar(self, visualizar=True):
        """
        Ejecuta el algoritmo PSO para encontrar el mínimo de la función objetivo.
        
        Parámetros:
        - visualizar: Si es True, muestra la evolución del enjambre en cada iteración
        
        Retorna:
        - La mejor posición encontrada y su valor de función
        """
        if visualizar:
            fig, ax = visualizar_funcion()
            # Graficar el punto mínimo real
            ax.scatter(1, 1, 0, color='green', s=100, label='Mínimo real')
        
        # Guardar posiciones iniciales
        posiciones_iter = [np.array([p['posicion'] for p in self.particulas])]
        
        # Iterar hasta alcanzar el máximo de iteraciones
        for iter in range(self.max_iter):
            for i, particula in enumerate(self.particulas):
                # Actualizar velocidad
                r1, r2 = random.random(), random.random()
                
                componente_inercia = self.w * particula['velocidad']
                componente_cognitivo = self.c1 * r1 * (particula['mejor_pos'] - particula['posicion'])
                componente_social = self.c2 * r2 * (self.mejor_global_pos - particula['posicion'])
                
                nueva_velocidad = componente_inercia + componente_cognitivo + componente_social
                
                # Actualizar posición
                nueva_posicion = particula['posicion'] + nueva_velocidad
                
                # Aplicar límites
                nueva_posicion[0] = max(self.limites[0], min(self.limites[1], nueva_posicion[0]))
                nueva_posicion[1] = max(self.limites[2], min(self.limites[3], nueva_posicion[1]))
                
                # Evaluar nueva posición
                nuevo_costo = funcion_objetivo(nueva_posicion[0], nueva_posicion[1])
                
                # Actualizar partícula
                particula['posicion'] = nueva_posicion
                particula['velocidad'] = nueva_velocidad
                particula['costo'] = nuevo_costo
                
                # Actualizar mejor posición personal
                if nuevo_costo < particula['mejor_costo']:
                    particula['mejor_pos'] = nueva_posicion.copy()
                    particula['mejor_costo'] = nuevo_costo
                    
                    # Actualizar mejor global
                    if nuevo_costo < self.mejor_global_costo:
                        self.mejor_global_costo = nuevo_costo
                        self.mejor_global_pos = nueva_posicion.copy()
            
            # Guardar posiciones de esta iteración
            posiciones_iter.append(np.array([p['posicion'] for p in self.particulas]))
            
            # Mostrar progreso
            if (iter + 1) % 10 == 0 or iter == 0:
                print(f"Iteración {iter+1}/{self.max_iter}, Mejor costo: {self.mejor_global_costo:.6f}")
                print(f"Mejor posición: [{self.mejor_global_pos[0]:.6f}, {self.mejor_global_pos[1]:.6f}]")
        
        # Visualizar la evolución del enjambre
        if visualizar:
            self.visualizar_evolucion(posiciones_iter, fig, ax)
        
        return self.mejor_global_pos, self.mejor_global_costo
    
    def visualizar_evolucion(self, posiciones_iter, fig, ax):
        """
        Visualiza la evolución del enjambre a lo largo de las iteraciones.
        
        Parámetros:
        - posiciones_iter: Lista de arrays con las posiciones de las partículas en cada iteración
        - fig, ax: Figura y ejes de matplotlib para la visualización
        """
        # Graficar la posición final del enjambre
        pos_finales = posiciones_iter[-1]
        ax.scatter(pos_finales[:, 0], pos_finales[:, 1], 
                  funcion_objetivo(pos_finales[:, 0], pos_finales[:, 1]),
                  color='red', s=50, label='Partículas finales')
        
        # Graficar la mejor posición encontrada
        ax.scatter(self.mejor_global_pos[0], self.mejor_global_pos[1], 
                  self.mejor_global_costo, color='blue', s=100, 
                  label='Mejor solución')
        
        # Añadir leyenda
        ax.legend()
        
        # Crear una animación de la evolución
        plt.figure(figsize=(10, 8))
        
        # Graficar contorno de la función
        x = np.linspace(self.limites[0], self.limites[1], 100)
        y = np.linspace(self.limites[2], self.limites[3], 100)
        X, Y = np.meshgrid(x, y)
        Z = funcion_objetivo(X, Y)
        
        plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.8)
        plt.colorbar(label='f(x,y)')
        
        # Graficar trayectoria de algunas partículas
        colores = plt.cm.jet(np.linspace(0, 1, self.num_particulas))
        
        for i in range(self.num_particulas):
            trayectoria = np.array([pos[i] for pos in posiciones_iter])
            plt.plot(trayectoria[:, 0], trayectoria[:, 1], '-', color=colores[i], alpha=0.5)
            plt.plot(trayectoria[-1, 0], trayectoria[-1, 1], 'o', color=colores[i])
        
        # Graficar el mínimo real y el encontrado
        plt.plot(1, 1, 'g*', markersize=15, label='Mínimo real')
        plt.plot(self.mejor_global_pos[0], self.mejor_global_pos[1], 'b*', 
                markersize=15, label='Mejor solución')
        
        plt.title('Trayectoria de las partículas durante la optimización')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('evolucion_pso.png')
        plt.show()

# Ejecutar el algoritmo
if __name__ == "__main__":
    print("Optimización por Enjambre de Partículas (PSO) para la función de Rosenbrock")
    print("=" * 80)
    
    # Parámetros del algoritmo
    num_particulas = 30
    max_iter = 100
    limites = [-2, 2, -1, 3]  # [min_x, max_x, min_y, max_y]
    
    # Crear y ejecutar el optimizador
    inicio = time.time()
    pso = PSO(num_particulas=num_particulas, max_iter=max_iter, limites=limites)
    mejor_pos, mejor_costo = pso.optimizar()
    fin = time.time()
    
    # Mostrar resultados
    print("\nResultados finales:")
    print(f"Mejor posición encontrada: [{mejor_pos[0]:.6f}, {mejor_pos[1]:.6f}]")
    print(f"Valor de la función en el mínimo: {mejor_costo:.10f}")
    print(f"Posición del mínimo real: [1.0, 1.0]")
    print(f"Valor de la función en el mínimo real: {funcion_objetivo(1, 1)}")
    print(f"Error absoluto: {np.sqrt((mejor_pos[0]-1)**2 + (mejor_pos[1]-1)**2):.10f}")
    print(f"Tiempo de ejecución: {fin - inicio:.2f} segundos")

