import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class ScatterSearch:
    """
    Implementación del algoritmo Scatter Search para el problema de la mochila.
    
    El algoritmo Scatter Search es una metaheurística evolutiva que combina soluciones
    de un conjunto de referencia para generar nuevas soluciones, aplicando mejoras
    locales y manteniendo diversidad en la población.
    
    Atributos:
        pesos (list): Lista de pesos de cada ítem.
        valores (list): Lista de valores de cada ítem.
        capacidad_maxima (float): Capacidad máxima de la mochila.
        n_items (int): Número total de ítems disponibles.
        tam_poblacion (int): Tamaño de la población inicial.
        tam_refset (int): Tamaño del conjunto de referencia.
        max_iter (int): Número máximo de iteraciones.
        diversificacion_iter (int): Frecuencia de diversificación.
        mejor_solucion (list): Mejor solución encontrada.
        mejor_valor (float): Valor de la mejor solución encontrada.
        historial_fitness (list): Historial de los mejores valores en cada iteración.
    """
    def __init__(self, pesos, valores, capacidad_maxima, tam_poblacion=10, tam_refset=5, 
                 max_iter=100, diversificacion_iter=20):
        """
        Inicializa el algoritmo Scatter Search con los parámetros del problema.
        
        Args:
            pesos (list): Lista de pesos de cada ítem.
            valores (list): Lista de valores de cada ítem.
            capacidad_maxima (float): Capacidad máxima de la mochila.
            tam_poblacion (int, opcional): Tamaño de la población inicial. Por defecto 10.
            tam_refset (int, opcional): Tamaño del conjunto de referencia. Por defecto 5.
            max_iter (int, opcional): Número máximo de iteraciones. Por defecto 100.
            diversificacion_iter (int, opcional): Frecuencia de diversificación. Por defecto 20.
        """
        self.pesos = pesos
        self.valores = valores
        self.capacidad_maxima = capacidad_maxima
        self.n_items = len(pesos)
        self.tam_poblacion = tam_poblacion
        self.tam_refset = tam_refset
        self.max_iter = max_iter
        self.diversificacion_iter = diversificacion_iter
        self.mejor_solucion = None
        self.mejor_valor = 0
        self.historial_fitness = []
    
    def generar_solucion_aleatoria(self):
        """
        Genera una solución aleatoria para el problema de la mochila.
        
        Cada solución es un vector binario donde 1 indica que el ítem está en la mochila
        y 0 que no está incluido.
        
        Returns:
            list: Vector binario que representa una solución aleatoria.
        """
        return [random.randint(0, 1) for _ in range(self.n_items)]
    
    def evaluar_solucion(self, solucion):
        """
        Evalúa una solución para el problema de la mochila.
        
        Calcula el valor total de los ítems seleccionados. Si el peso total excede
        la capacidad máxima, la solución se considera inválida y devuelve 0.
        
        Args:
            solucion (list): Vector binario que representa una solución.
            
        Returns:
            float: Valor total de la solución, o 0 si excede la capacidad.
        """
        peso_total = sum(self.pesos[i] * solucion[i] for i in range(self.n_items))
        valor_total = sum(self.valores[i] * solucion[i] for i in range(self.n_items))
        
        # Si excede la capacidad, penalizar
        if peso_total > self.capacidad_maxima:
            return 0
        return valor_total
    
    def distancia_hamming(self, sol1, sol2):
        """
        Calcula la distancia de Hamming entre dos soluciones.
        
        La distancia de Hamming cuenta el número de posiciones en las que
        los elementos correspondientes de dos vectores son diferentes.
        
        Args:
            sol1 (list): Primera solución.
            sol2 (list): Segunda solución.
            
        Returns:
            int: Distancia de Hamming entre las dos soluciones.
        """
        return sum(s1 != s2 for s1, s2 in zip(sol1, sol2))
    
    def generar_poblacion_diversa(self):
        """
        Genera una población diversa de soluciones aleatorias.
        
        Returns:
            list: Lista de tuplas (solucion, valor) donde solucion es un vector binario
                 y valor es el resultado de evaluar dicha solución.
        """
        poblacion = []
        for _ in range(self.tam_poblacion):
            solucion = self.generar_solucion_aleatoria()
            poblacion.append((solucion, self.evaluar_solucion(solucion)))
        return poblacion
    
    def actualizar_refset(self, refset, poblacion):
        """
        Actualiza el conjunto de referencia con las mejores soluciones y las más diversas.
        
        El conjunto de referencia se compone de dos partes:
        1. Las mejores soluciones según su valor.
        2. Las soluciones más diversas respecto a las ya seleccionadas.
        
        Args:
            refset (list): Conjunto de referencia actual.
            poblacion (list): Población de soluciones candidatas.
            
        Returns:
            list: Nuevo conjunto de referencia actualizado.
        """
        # Ordenar población por fitness
        poblacion_ordenada = sorted(poblacion, key=lambda x: x[1], reverse=True)
        
        # Seleccionar las mejores soluciones
        mejores = poblacion_ordenada[:self.tam_refset // 2]
        
        # Seleccionar soluciones diversas
        candidatos = poblacion_ordenada[self.tam_refset // 2:]
        diversas = []
        
        # Añadir soluciones al refset basadas en diversidad
        soluciones_refset = [sol for sol, _ in mejores]
        
        while len(diversas) < self.tam_refset // 2 and candidatos:
            max_distancia = -1
            max_index = -1
            
            for i, (candidato, _) in enumerate(candidatos):
                # Calcular la distancia mínima a cualquier solución en el refset
                min_distancia = min(self.distancia_hamming(candidato, sol_ref) 
                                   for sol_ref in soluciones_refset) if soluciones_refset else float('inf')
                
                if min_distancia > max_distancia:
                    max_distancia = min_distancia
                    max_index = i
            
            if max_index != -1:
                diversas.append(candidatos.pop(max_index))
                soluciones_refset.append(diversas[-1][0])
        
        return mejores + diversas
    
    def combinar_soluciones(self, sol1, sol2):
        """
        Combina dos soluciones para generar una nueva mediante cruce de un punto.
        
        Args:
            sol1 (list): Primera solución padre.
            sol2 (list): Segunda solución padre.
            
        Returns:
            list: Nueva solución generada por combinación.
        """
        # Operador de cruce de un punto
        punto_cruce = random.randint(1, self.n_items - 1)
        nueva_sol = sol1[:punto_cruce] + sol2[punto_cruce:]
        return nueva_sol
    
    def mejorar_solucion(self, solucion):
        """
        Aplica una búsqueda local para mejorar la solución.
        
        Explora el vecindario de la solución cambiando un bit a la vez (flip)
        y selecciona la mejor mejora encontrada.
        
        Args:
            solucion (list): Solución a mejorar.
            
        Returns:
            tuple: (mejor_solucion, mejor_valor) donde mejor_solucion es la solución mejorada
                  y mejor_valor es su valor asociado.
        """
        mejor_sol = solucion.copy()
        mejor_valor = self.evaluar_solucion(mejor_sol)
        
        # Explorar vecindario (flip de un bit)
        for i in range(self.n_items):
            nueva_sol = mejor_sol.copy()
            nueva_sol[i] = 1 - nueva_sol[i]  # Cambiar 0 a 1 o 1 a 0
            nuevo_valor = self.evaluar_solucion(nueva_sol)
            
            if nuevo_valor > mejor_valor:
                mejor_sol = nueva_sol
                mejor_valor = nuevo_valor
        
        return mejor_sol, mejor_valor
    
    def generar_combinaciones(self, refset):
        """
        Genera todas las combinaciones posibles de pares del conjunto de referencia.
        
        Args:
            refset (list): Conjunto de referencia actual.
            
        Returns:
            list: Lista de tuplas (sol1, sol2) representando todas las combinaciones posibles.
        """
        combinaciones = []
        soluciones = [sol for sol, _ in refset]
        
        for i in range(len(soluciones)):
            for j in range(i + 1, len(soluciones)):
                combinaciones.append((soluciones[i], soluciones[j]))
        
        return combinaciones
    
    def ejecutar(self):
        """
        Ejecuta el algoritmo Scatter Search completo.
        
        El algoritmo sigue estos pasos:
        1. Generar población inicial diversa
        2. Crear conjunto de referencia inicial
        3. Para cada iteración:
           a. Generar combinaciones de soluciones
           b. Crear nuevas soluciones mediante combinación
           c. Mejorar las nuevas soluciones
           d. Actualizar el conjunto de referencia
           e. Actualizar la mejor solución global
           f. Aplicar diversificación periódica
        """
        # Generar población inicial
        poblacion = self.generar_poblacion_diversa()
        
        # Inicializar conjunto de referencia
        refset = self.actualizar_refset([], poblacion)
        
        for iteracion in range(self.max_iter):
            # Generar combinaciones
            combinaciones = self.generar_combinaciones(refset)
            
            # Generar nuevas soluciones
            nuevas_soluciones = []
            for sol1, sol2 in combinaciones:
                nueva_sol = self.combinar_soluciones(sol1, sol2)
                nueva_sol_mejorada, valor = self.mejorar_solucion(nueva_sol)
                nuevas_soluciones.append((nueva_sol_mejorada, valor))
            
            # Actualizar conjunto de referencia
            todas_soluciones = refset + nuevas_soluciones
            refset = self.actualizar_refset(refset, todas_soluciones)
            
            # Actualizar mejor solución
            mejor_actual = max(refset, key=lambda x: x[1])
            if mejor_actual[1] > self.mejor_valor:
                self.mejor_solucion = mejor_actual[0]
                self.mejor_valor = mejor_actual[1]
            
            # Guardar historial
            self.historial_fitness.append(self.mejor_valor)
            
            # Diversificación periódica
            if iteracion > 0 and iteracion % self.diversificacion_iter == 0:
                nueva_poblacion = self.generar_poblacion_diversa()
                refset = self.actualizar_refset(refset, nueva_poblacion)
            
            # Imprimir progreso
            if (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}/{self.max_iter}, Mejor valor: {self.mejor_valor}")
    
    def mostrar_resultados(self):
        """
        Muestra los resultados del algoritmo y genera una gráfica de convergencia.
        
        Presenta:
        - El mejor valor encontrado
        - El peso total de la solución
        - Los ítems seleccionados con sus pesos y valores
        - Una gráfica de la evolución del fitness a lo largo de las iteraciones
        """
        print("\nResultados finales:")
        print(f"Mejor valor encontrado: {self.mejor_valor}")
        
        peso_total = sum(self.pesos[i] * self.mejor_solucion[i] for i in range(self.n_items))
        print(f"Peso total: {peso_total}/{self.capacidad_maxima}")
        
        print("Items seleccionados:")
        for i in range(self.n_items):
            if self.mejor_solucion[i] == 1:
                print(f"Item {i+1}: Peso={self.pesos[i]}, Valor={self.valores[i]}")
        
        # Graficar convergencia
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historial_fitness)), self.historial_fitness, 'b-')
        plt.title('Convergencia del algoritmo Scatter Search')
        plt.xlabel('Iteración')
        plt.ylabel('Valor de la mochila')
        plt.grid(True)
        plt.show()


# Ejemplos de uso con diferentes casos
if __name__ == "__main__":
    print("CASO 1: Problema con muchos ítems de bajo valor/peso")
    # Muchos ítems pequeños, algunos con alto valor
    pesos = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 3]
    valores = [1, 2, 1, 3, 2, 1, 2, 1, 3, 2, 5, 10, 2, 4, 1, 2, 1, 3, 2, 8]
    capacidad_maxima = 10
    
    scatter1 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=100,
        tam_refset=15,
        max_iter=80,
        diversificacion_iter=15
    )
    
    print("Ejecutando Scatter Search para el problema de la mochila con muchos ítems pequeños...")
    scatter1.ejecutar()
    scatter1.mostrar_resultados()
    
    print("\n" + "="*50 + "\n")
    
    print("CASO 2: Problema con ítems de alto valor pero gran peso")
    # Pocos ítems valiosos pero pesados
    pesos = [10, 15, 20, 8, 12, 5, 18, 25, 15, 10]
    valores = [50, 60, 90, 30, 40, 10, 70, 100, 55, 35]
    capacidad_maxima = 35
    
    scatter2 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=80,
        tam_refset=12,
        max_iter=150,
        diversificacion_iter=25
    )
    
    print("Ejecutando Scatter Search para el problema con ítems valiosos pero pesados...")
    scatter2.ejecutar()
    scatter2.mostrar_resultados()
    
    print("\n" + "="*50 + "\n")
    
    print("CASO 3: Problema con valores y pesos correlacionados")
    # Valores proporcionales a los pesos (dificulta la decisión)
    pesos = [3, 5, 7, 10, 2, 4, 8, 6, 9, 12, 15, 5, 8, 4, 7]
    valores = [6, 10, 14, 20, 4, 8, 16, 12, 18, 24, 30, 10, 16, 8, 14]
    capacidad_maxima = 30
    
    scatter3 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=120,
        tam_refset=20,
        max_iter=200,
        diversificacion_iter=30
    )
    
    print("Ejecutando Scatter Search para el problema con valores proporcionales a pesos...")
    scatter3.ejecutar()
    scatter3.mostrar_resultados()


