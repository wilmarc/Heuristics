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
    
    def ejecutar(self, verbose=False):
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
           
        Args:
            verbose (bool): Si es True, muestra información detallada del proceso.
        """
        # Generar población inicial
        poblacion = self.generar_poblacion_diversa()
        
        if verbose:
            print("\n--- INICIO DEL ALGORITMO ---")
            print(f"Población inicial generada: {self.tam_poblacion} soluciones")
        
        # Inicializar conjunto de referencia
        refset = self.actualizar_refset([], poblacion)
        
        if verbose:
            print(f"Conjunto de referencia inicial creado con {len(refset)} soluciones")
            print(f"Mejor solución inicial: {max(refset, key=lambda x: x[1])[1]}")
        
        for iteracion in range(self.max_iter):
            if verbose and iteracion > 0 and iteracion % 10 == 0:
                print(f"\n--- ITERACIÓN {iteracion} ---")
            
            # Generar combinaciones
            combinaciones = self.generar_combinaciones(refset)
            
            if verbose and iteracion % 10 == 0:
                print(f"Generadas {len(combinaciones)} combinaciones de soluciones")
            
            # Generar nuevas soluciones
            nuevas_soluciones = []
            for sol1, sol2 in combinaciones:
                nueva_sol = self.combinar_soluciones(sol1, sol2)
                nueva_sol_mejorada, valor = self.mejorar_solucion(nueva_sol)
                nuevas_soluciones.append((nueva_sol_mejorada, valor))
            
            if verbose and iteracion % 10 == 0:
                print(f"Creadas y mejoradas {len(nuevas_soluciones)} nuevas soluciones")
            
            # Actualizar conjunto de referencia
            todas_soluciones = refset + nuevas_soluciones
            refset_anterior = refset.copy()
            refset = self.actualizar_refset(refset, todas_soluciones)
            
            # Calcular cuántas soluciones nuevas entraron al refset
            soluciones_anteriores = set(tuple(sol) for sol, _ in refset_anterior)
            soluciones_actuales = set(tuple(sol) for sol, _ in refset)
            nuevas_en_refset = len(soluciones_actuales - soluciones_anteriores)
            
            if verbose and iteracion % 10 == 0:
                print(f"Conjunto de referencia actualizado: {nuevas_en_refset} nuevas soluciones incorporadas")
            
            # Actualizar mejor solución
            mejor_actual = max(refset, key=lambda x: x[1])
            if mejor_actual[1] > self.mejor_valor:
                self.mejor_solucion = mejor_actual[0]
                self.mejor_valor = mejor_actual[1]
                
                if verbose:
                    print(f"¡Nueva mejor solución encontrada! Valor: {self.mejor_valor}")
            
            # Guardar historial
            self.historial_fitness.append(self.mejor_valor)
            
            # Diversificación periódica
            if iteracion > 0 and iteracion % self.diversificacion_iter == 0:
                nueva_poblacion = self.generar_poblacion_diversa()
                refset = self.actualizar_refset(refset, nueva_poblacion)
                
                if verbose:
                    print(f"Aplicada diversificación: generadas {self.tam_poblacion} nuevas soluciones")
            
            # Imprimir progreso
            if (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}/{self.max_iter}, Mejor valor: {self.mejor_valor}")
    
    def mostrar_resultados(self, titulo="Resultados del algoritmo Scatter Search", comparar_con_optimo=None):
        """
        Muestra los resultados del algoritmo y genera una gráfica de convergencia.
        
        Presenta:
        - El mejor valor encontrado
        - El peso total de la solución
        - Los ítems seleccionados con sus pesos y valores
        - Una gráfica de la evolución del fitness a lo largo de las iteraciones
        
        Args:
            titulo (str): Título para la gráfica de convergencia
            comparar_con_optimo (tuple, opcional): Tupla (valor_optimo, solucion_optima) para comparar
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
        plt.plot(range(len(self.historial_fitness)), self.historial_fitness, 'b-', label='Scatter Search')
        
        # Si hay un valor óptimo para comparar, mostrarlo
        if comparar_con_optimo:
            valor_optimo, _ = comparar_con_optimo
            plt.axhline(y=valor_optimo, color='r', linestyle='--', label=f'Valor óptimo: {valor_optimo}')
            
            # Calcular gap porcentual
            gap = ((valor_optimo - self.mejor_valor) / valor_optimo) * 100 if valor_optimo > 0 else 0
            plt.text(len(self.historial_fitness) * 0.7, self.mejor_valor, 
                     f'Gap: {gap:.2f}%', fontsize=10, verticalalignment='bottom')
        
        plt.title(titulo)
        plt.xlabel('Iteración')
        plt.ylabel('Valor de la mochila')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Si hay un valor óptimo, mostrar comparación detallada
        if comparar_con_optimo:
            valor_optimo, solucion_optima = comparar_con_optimo
            
            print("\nComparación con solución óptima:")
            print(f"Valor óptimo: {valor_optimo}")
            print(f"Valor encontrado: {self.mejor_valor}")
            print(f"Gap: {((valor_optimo - self.mejor_valor) / valor_optimo) * 100 if valor_optimo > 0 else 0:.2f}%")
            
            # Comparar soluciones
            print("\nComparación de soluciones:")
            print(f"{'Ítem':<5} {'Peso':<6} {'Valor':<6} {'Óptima':<8} {'Encontrada':<10}")
            print("-" * 40)
            for i in range(self.n_items):
                print(f"{i+1:<5} {self.pesos[i]:<6} {self.valores[i]:<6} {solucion_optima[i]:<8} {self.mejor_solucion[i]:<10}")
            
            # Visualizar diferencias
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Gráfico de barras para comparar valores
            items = [f"Ítem {i+1}" for i in range(self.n_items)]
            valores_optimos = [self.valores[i] * solucion_optima[i] for i in range(self.n_items)]
            valores_encontrados = [self.valores[i] * self.mejor_solucion[i] for i in range(self.n_items)]
            
            x = np.arange(len(items))
            width = 0.35
            
            ax1.bar(x - width/2, valores_optimos, width, label='Solución Óptima')
            ax1.bar(x + width/2, valores_encontrados, width, label='Solución Encontrada')
            
            ax1.set_title('Comparación de valores por ítem')
            ax1.set_xlabel('Ítems')
            ax1.set_ylabel('Valor')
            ax1.set_xticks(x)
            ax1.set_xticklabels(items, rotation=45)
            ax1.legend()
            
            # Gráfico de pie para comparar composición
            labels = ['Ítems comunes', 'Solo en óptima', 'Solo en encontrada']
            comunes = sum(1 for i in range(self.n_items) if solucion_optima[i] == 1 and self.mejor_solucion[i] == 1)
            solo_optima = sum(1 for i in range(self.n_items) if solucion_optima[i] == 1 and self.mejor_solucion[i] == 0)
            solo_encontrada = sum(1 for i in range(self.n_items) if solucion_optima[i] == 0 and self.mejor_solucion[i] == 1)
            
            sizes = [comunes, solo_optima, solo_encontrada]
            colors = ['lightgreen', 'lightcoral', 'lightskyblue']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            ax2.set_title('Comparación de composición de soluciones')
            
            plt.tight_layout()
            plt.show()


def demostrar_scatter_search_paso_a_paso():
    """
    Demuestra el algoritmo Scatter Search paso a paso, mostrando cada etapa del proceso
    para que los estudiantes puedan visualizar cómo funciona el algoritmo internamente.
    """
    print("\n" + "="*80)
    print("DEMOSTRACIÓN PASO A PASO DEL ALGORITMO SCATTER SEARCH".center(80))
    print("="*80)
    
    # Definir un problema pequeño para facilitar la visualización
    pesos = [2, 3, 4, 5, 9]
    valores = [3, 4, 5, 8, 10]
    capacidad_maxima = 10
    
    print("\nPROBLEMA DE LA MOCHILA:")
    print("-"*50)
    print(f"Capacidad máxima: {capacidad_maxima}")
    print("Ítems disponibles:")
    for i in range(len(pesos)):
        print(f"  Ítem {i+1}: Peso={pesos[i]}, Valor={valores[i]}, Ratio={valores[i]/pesos[i]:.2f}")
    
    # Crear instancia con parámetros pequeños para facilitar la visualización
    ss = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=8,    # Población pequeña para visualización
        tam_refset=4,       # RefSet pequeño para visualización
        max_iter=3,         # Pocas iteraciones para el ejemplo
        diversificacion_iter=2
    )
    
    print("\nPARÁMETROS DEL ALGORITMO:")
    print("-"*50)
    print(f"Tamaño de población: {ss.tam_poblacion}")
    print(f"Tamaño del conjunto de referencia: {ss.tam_refset}")
    print(f"Número de iteraciones: {ss.max_iter}")
    print(f"Frecuencia de diversificación: {ss.diversificacion_iter}")
    
    # PASO 1: Generar población inicial
    print("\n" + "="*50)
    print("PASO 1: GENERACIÓN DE POBLACIÓN INICIAL")
    print("="*50)
    
    poblacion_inicial = ss.generar_poblacion_diversa()
    
    print("Población inicial generada:")
    print("-"*80)
    print(f"{'#':<3} {'Solución':<20} {'Peso':<8} {'Valor':<8} {'Factible':<10}")
    print("-"*80)
    
    for i, (solucion, valor) in enumerate(poblacion_inicial):
        peso = sum(ss.pesos[j] * solucion[j] for j in range(ss.n_items))
        factible = "Sí" if peso <= ss.capacidad_maxima else "No"
        print(f"{i+1:<3} {str(solucion):<20} {peso:<8} {valor:<8} {factible:<10}")
    
    # PASO 2: Crear conjunto de referencia inicial
    print("\n" + "="*50)
    print("PASO 2: CREACIÓN DEL CONJUNTO DE REFERENCIA INICIAL")
    print("="*50)
    
    print("El conjunto de referencia se divide en dos partes:")
    print("1. Las mejores soluciones según su valor")
    print("2. Las soluciones más diversas respecto a las ya seleccionadas")
    
    refset = ss.actualizar_refset([], poblacion_inicial)
    
    # Identificar cuáles son las mejores y cuáles las diversas
    poblacion_ordenada = sorted(poblacion_inicial, key=lambda x: x[1], reverse=True)
    mejores = poblacion_ordenada[:ss.tam_refset // 2]
    mejores_indices = [i for i, (sol, _) in enumerate(poblacion_inicial) if any(np.array_equal(sol, m_sol) for m_sol, _ in mejores)]
    
    print("\nConjunto de referencia inicial:")
    print("-"*80)
    print(f"{'#':<3} {'Solución':<20} {'Peso':<8} {'Valor':<8} {'Tipo':<10}")
    print("-"*80)
    
    for i, (solucion, valor) in enumerate(refset):
        peso = sum(ss.pesos[j] * solucion[j] for j in range(ss.n_items))
        # Determinar si es de las mejores o de las diversas
        tipo = "Mejor" if any(np.array_equal(solucion, m_sol) for m_sol, _ in mejores) else "Diversa"
        print(f"{i+1:<3} {str(solucion):<20} {peso:<8} {valor:<8} {tipo:<10}")
    
    # PASO 3: Iteraciones del algoritmo
    for iteracion in range(ss.max_iter):
        print("\n" + "="*50)
        print(f"ITERACIÓN {iteracion+1} DEL ALGORITMO")
        print("="*50)
        
        # PASO 3.1: Generar combinaciones
        print("\nPASO 3.1: GENERACIÓN DE COMBINACIONES")
        print("-"*50)
        
        combinaciones = ss.generar_combinaciones(refset)
        
        print(f"Se generan {len(combinaciones)} combinaciones posibles entre las soluciones del RefSet:")
        for i, (sol1, sol2) in enumerate(combinaciones):
            print(f"Combinación {i+1}: {sol1} + {sol2}")
        
        # PASO 3.2: Crear nuevas soluciones mediante combinación
        print("\nPASO 3.2: CREACIÓN DE NUEVAS SOLUCIONES MEDIANTE COMBINACIÓN")
        print("-"*50)
        
        nuevas_soluciones = []
        print("Para cada combinación, se aplica un operador de cruce y luego mejora local:")
        
        for i, (sol1, sol2) in enumerate(combinaciones):
            # Mostrar el proceso de combinación
            punto_cruce = random.randint(1, ss.n_items - 1)
            nueva_sol = sol1[:punto_cruce] + sol2[punto_cruce:]
            
            print(f"\nCombinación {i+1}:")
            print(f"Padre 1: {sol1}")
            print(f"Padre 2: {sol2}")
            print(f"Punto de cruce: {punto_cruce}")
            print(f"Hijo antes de mejora: {nueva_sol}")
            
            # Aplicar mejora local
            nueva_sol_mejorada, valor = ss.mejorar_solucion(nueva_sol)
            peso = sum(ss.pesos[j] * nueva_sol_mejorada[j] for j in range(ss.n_items))
            
            print(f"Hijo después de mejora: {nueva_sol_mejorada}")
            print(f"Valor: {valor}, Peso: {peso}")
            
            nuevas_soluciones.append((nueva_sol_mejorada, valor))
        
        # PASO 3.3: Actualizar conjunto de referencia
        print("\nPASO 3.3: ACTUALIZACIÓN DEL CONJUNTO DE REFERENCIA")
        print("-"*50)
        
        print("Todas las soluciones candidatas (RefSet actual + nuevas soluciones):")
        todas_soluciones = refset + nuevas_soluciones
        
        print("-"*80)
        print(f"{'#':<3} {'Solución':<20} {'Peso':<8} {'Valor':<8} {'Origen':<10}")
        print("-"*80)
        
        for i, (solucion, valor) in enumerate(todas_soluciones):
            peso = sum(ss.pesos[j] * solucion[j] for j in range(ss.n_items))
            origen = "RefSet" if i < len(refset) else "Nueva"
            print(f"{i+1:<3} {str(solucion):<20} {peso:<8} {valor:<8} {origen:<10}")
        
        # Guardar el RefSet anterior para comparación
        refset_anterior = refset.copy()
        
        # Actualizar el RefSet
        refset = ss.actualizar_refset(refset, todas_soluciones)
        
        # Identificar qué soluciones entraron y salieron del RefSet
        soluciones_anteriores = [tuple(sol) for sol, _ in refset_anterior]
        soluciones_actuales = [tuple(sol) for sol, _ in refset]
        
        nuevas_en_refset = [sol for sol in soluciones_actuales if sol not in soluciones_anteriores]
        salieron_de_refset = [sol for sol in soluciones_anteriores if sol not in soluciones_actuales]
        
        print("\nNuevo conjunto de referencia:")
        print("-"*80)
        print(f"{'#':<3} {'Solución':<20} {'Peso':<8} {'Valor':<8} {'Estado':<10}")
        print("-"*80)
        
        for i, (solucion, valor) in enumerate(refset):
            peso = sum(ss.pesos[j] * solucion[j] for j in range(ss.n_items))
            estado = "Nueva" if tuple(solucion) in nuevas_en_refset else "Mantenida"
            print(f"{i+1:<3} {str(solucion):<20} {peso:<8} {valor:<8} {estado:<10}")
        
        if salieron_de_refset:
            print("\nSoluciones que salieron del RefSet:")
            for sol in salieron_de_refset:
                print(f"Solución: {list(sol)}")
        
        # PASO 3.4: Actualizar mejor solución global
        mejor_actual = max(refset, key=lambda x: x[1])
        if mejor_actual[1] > ss.mejor_valor:
            ss.mejor_solucion = mejor_actual[0]
            ss.mejor_valor = mejor_actual[1]
            print(f"\n¡Nueva mejor solución encontrada! Valor: {ss.mejor_valor}")
        else:
            print(f"\nMejor solución actual: Valor = {ss.mejor_valor}")
        
        # Guardar historial
        ss.historial_fitness.append(ss.mejor_valor)
        
        # PASO 3.5: Diversificación periódica
        if iteracion > 0 and (iteracion + 1) % ss.diversificacion_iter == 0:
            print("\nPASO 3.5: DIVERSIFICACIÓN PERIÓDICA")
            print("-"*50)
            
            print("Generando nueva población para diversificar la búsqueda...")
            nueva_poblacion = ss.generar_poblacion_diversa()
            
            print(f"Nueva población generada con {len(nueva_poblacion)} soluciones")
            refset = ss.actualizar_refset(refset, nueva_poblacion)
            
            print("Conjunto de referencia actualizado después de la diversificación")
    
    # Mostrar resultados finales
    print("\n" + "="*50)
    print("RESULTADOS FINALES DEL ALGORITMO")
    print("="*50)
    
    print(f"Mejor valor encontrado: {ss.mejor_valor}")
    
    peso_total = sum(ss.pesos[i] * ss.mejor_solucion[i] for i in range(ss.n_items))
    print(f"Peso total: {peso_total}/{ss.capacidad_maxima}")
    
    print("Items seleccionados:")
    for i in range(ss.n_items):
        if ss.mejor_solucion[i] == 1:
            print(f"Item {i+1}: Peso={ss.pesos[i]}, Valor={ss.valores[i]}")
    
    # Graficar convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ss.historial_fitness)), ss.historial_fitness, 'b-o')
    plt.title("Convergencia del algoritmo Scatter Search")
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la mochila')
    plt.grid(True)
    plt.show()
    
    print("\n" + "="*80)
    print("FIN DE LA DEMOSTRACIÓN PASO A PASO".center(80))
    print("="*80)


def visualizar_componentes_scatter_search():
    """
    Visualiza los componentes clave del algoritmo Scatter Search con ejemplos
    específicos para cada uno, facilitando la comprensión de su funcionamiento.
    """
    print("\n" + "="*80)
    print("VISUALIZACIÓN DE COMPONENTES DEL SCATTER SEARCH".center(80))
    print("="*80)
    
    # Definir un problema pequeño para facilitar la visualización
    pesos = [2, 3, 4, 5, 9]
    valores = [3, 4, 5, 8, 10]
    capacidad_maxima = 10
    
    print("\nPROBLEMA DE LA MOCHILA:")
    print("-"*50)
    print(f"Capacidad máxima: {capacidad_maxima}")
    print("Ítems disponibles:")
    for i in range(len(pesos)):
        print(f"  Ítem {i+1}: Peso={pesos[i]}, Valor={valores[i]}, Ratio={valores[i]/pesos[i]:.2f}")
    
    # Crear instancia
    ss = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=8,
        tam_refset=4,
        max_iter=3,
        diversificacion_iter=2
    )
    
    # 1. VISUALIZACIÓN DE GENERACIÓN DE SOLUCIONES
    print("\n" + "="*50)
    print("1. GENERACIÓN DE SOLUCIONES ALEATORIAS")
    print("="*50)
    
    print("Generando 5 soluciones aleatorias:")
    for i in range(5):
        solucion = ss.generar_solucion_aleatoria()
        valor = ss.evaluar_solucion(solucion)
        peso = sum(ss.pesos[j] * solucion[j] for j in range(ss.n_items))
        factible = "Sí" if peso <= ss.capacidad_maxima else "No"
        
        print(f"\nSolución {i+1}: {solucion}")
    print("Parámetros del algoritmo:")
    print(f"- Tamaño de población: {ss.tam_poblacion}")
    print(f"- Tamaño del conjunto de referencia: {ss.tam_refset}")
    print(f"- Máximo de iteraciones: {ss.max_iter}")
    print(f"- Frecuencia de diversificación: {ss.diversificacion_iter}")
    
    print("\nEjecutando algoritmo con información detallada...")
    ss.ejecutar(verbose=True)
    ss.mostrar_resultados("Convergencia del algoritmo - Caso 1 (Parámetros básicos)")
    
    # Segunda ejecución con parámetros mejorados
    print("\nSEGUNDA EJECUCIÓN: Parámetros mejorados")
    print("-"*60)
    
    scatter_demo2 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=40,  # Mayor población
        tam_refset=10,     # Mayor conjunto de referencia
        max_iter=50,       # Más iteraciones
        diversificacion_iter=15  # Diferente frecuencia de diversificación
    )
    
    print("Parámetros del algoritmo:")
    print(f"- Tamaño de población: {scatter_demo2.tam_poblacion}")
    print(f"- Tamaño del conjunto de referencia: {scatter_demo2.tam_refset}")
    print(f"- Máximo de iteraciones: {scatter_demo2.max_iter}")
    print(f"- Frecuencia de diversificación: {scatter_demo2.diversificacion_iter}")
    
    print("\nEjecutando algoritmo con información detallada...")
    scatter_demo2.ejecutar(verbose=True)
    scatter_demo2.mostrar_resultados("Convergencia del algoritmo - Caso 1 (Parámetros mejorados)")
    
    # Comparación de resultados
    print("\nCOMPARACIÓN DE RESULTADOS:")
    print("-"*60)
    print(f"Ejecución 1 - Mejor valor: {ss.mejor_valor}")
    print(f"Ejecución 2 - Mejor valor: {scatter_demo2.mejor_valor}")
    
    # Graficar comparación
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(ss.historial_fitness)), ss.historial_fitness, 'b-', label='Ejecución 1 (Parámetros básicos)')
    plt.plot(range(len(scatter_demo2.historial_fitness)), scatter_demo2.historial_fitness, 'r-', label='Ejecución 2 (Parámetros mejorados)')
    plt.title('Comparación de convergencia entre ejecuciones - Caso 1')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la mochila')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # CASO DEMOSTRATIVO 2: Problema más complejo
    print("\n" + "="*80)
    print("CASO DEMOSTRATIVO 2: Problema más complejo con diferentes configuraciones")
    print("="*80)
    
    # Problema más complejo
    pesos = [10, 15, 20, 8, 12, 5, 18, 25, 15, 10]
    valores = [50, 60, 90, 30, 40, 10, 70, 100, 55, 35]
    capacidad_maxima = 35
    
    print("Descripción del problema:")
    print(f"- Número de ítems: {len(pesos)}")
    print(f"- Capacidad de la mochila: {capacidad_maxima}")
    print("- Ítems disponibles:")
    for i in range(len(pesos)):
        print(f"  * Ítem {i+1}: Peso={pesos[i]}, Valor={valores[i]}, Ratio={valores[i]/pesos[i]:.2f}")
    
    # Primera ejecución con diversificación baja
    print("\nPRIMERA EJECUCIÓN: Diversificación baja")
    print("-"*60)
    
    scatter_demo3 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=50,
        tam_refset=10,
        max_iter=80,
        diversificacion_iter=40  # Diversificación poco frecuente
    )
    
    print("Parámetros del algoritmo:")
    print(f"- Tamaño de población: {scatter_demo3.tam_poblacion}")
    print(f"- Tamaño del conjunto de referencia: {scatter_demo3.tam_refset}")
    print(f"- Máximo de iteraciones: {scatter_demo3.max_iter}")
    print(f"- Frecuencia de diversificación: {scatter_demo3.diversificacion_iter}")
    
    print("\nEjecutando algoritmo con diversificación baja...")
    scatter_demo3.ejecutar(verbose=True)
    scatter_demo3.mostrar_resultados("Convergencia del algoritmo - Caso 2 (Diversificación baja)")
    
    # Segunda ejecución con diversificación alta
    print("\nSEGUNDA EJECUCIÓN: Diversificación alta")
    print("-"*60)
    
    scatter_demo4 = ScatterSearch(
        pesos=pesos,
        valores=valores,
        capacidad_maxima=capacidad_maxima,
        tam_poblacion=50,
        tam_refset=10,
        max_iter=80,
        diversificacion_iter=10  # Diversificación más frecuente
    )
    
    print("Parámetros del algoritmo:")
    print(f"- Tamaño de población: {scatter_demo4.tam_poblacion}")
    print(f"- Tamaño del conjunto de referencia: {scatter_demo4.tam_refset}")
    print(f"- Máximo de iteraciones: {scatter_demo4.max_iter}")
    print(f"- Frecuencia de diversificación: {scatter_demo4.diversificacion_iter}")
    
    print("\nEjecutando algoritmo con diversificación alta...")
    scatter_demo4.ejecutar(verbose=True)
    scatter_demo4.mostrar_resultados("Convergencia del algoritmo - Caso 2 (Diversificación alta)")
    
    # Comparación de resultados
    print("\nCOMPARACIÓN DE RESULTADOS:")
    print("-"*60)
    print(f"Ejecución 1 (Diversificación baja) - Mejor valor: {scatter_demo3.mejor_valor}")
    print(f"Ejecución 2 (Diversificación alta) - Mejor valor: {scatter_demo4.mejor_valor}")
    
    # Graficar comparación
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(scatter_demo3.historial_fitness)), scatter_demo3.historial_fitness, 'g-', label='Diversificación baja')
    plt.plot(range(len(scatter_demo4.historial_fitness)), scatter_demo4.historial_fitness, 'm-', label='Diversificación alta')
    plt.title('Comparación de convergencia entre estrategias de diversificación - Caso 2')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la mochila')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n" + "="*80)
    print("FIN DE LA DEMOSTRACIÓN SECUENCIAL".center(80))
    print("="*80)


def ejecutar_secuencia_demostrativa():
    """
    Ejecuta una secuencia demostrativa del algoritmo Scatter Search
    mostrando paso a paso su funcionamiento y visualizando sus componentes.
    """
    print("\nINICIANDO SECUENCIA DEMOSTRATIVA DEL ALGORITMO SCATTER SEARCH")
    print("="*80)
    
    # Ejecutar la demostración paso a paso
    demostrar_scatter_search_paso_a_paso()
    
    # Visualizar los componentes
    visualizar_componentes_scatter_search()


# Ejemplos de uso con diferentes casos
if __name__ == "__main__":
    # Ejecutar la secuencia demostrativa para estudiantes
    ejecutar_secuencia_demostrativa()
    
    print("\nEJEMPLOS ADICIONALES DE APLICACIÓN DEL ALGORITMO")
    print("="*50)
    
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