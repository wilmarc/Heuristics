import numpy as np
import networkx as nx
import random
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Configuración para mostrar gráficos 
plt.ion()  # Modo interactivo
plt.style.use('default')  # Usar estilo por defecto

# Configurar el backend de matplotlib
import matplotlib
matplotlib.use('TkAgg')  # Usar TkAgg como backend

class SolucionMinFlow:
    """
    Clase que representa una solución para el problema de flujo mínimo.
    
    Almacena los flujos asignados a cada arco, el grafo de la red, los nodos
    de origen y destino, así como información sobre la factibilidad y calidad
    de la solución.
    """
    def __init__(self, flujos, grafo, origen, destino):
        """
        Inicializa una solución para el problema de flujo mínimo.
        
        Args:
            flujos (dict): Diccionario que mapea arcos (u,v) a valores de flujo
            grafo (DiGraph): Grafo dirigido que representa la red
            origen (node): Nodo origen del flujo
            destino (node): Nodo destino del flujo
        """
        self.flujos = flujos  # Diccionario de flujos en cada arco
        self.grafo = grafo    # Grafo de la red
        self.origen = origen  # Nodo origen
        self.destino = destino  # Nodo destino
        self.valor_flujo = None  # Valor total del flujo
        self.factible = False  # Indica si la solución es factible
        self.calidad = float('inf')  # Medida de calidad (menor es mejor)
    
    def evaluar(self):
        """
        Evalúa la calidad de la solución y verifica su factibilidad.
        
        Comprueba la conservación de flujo en cada nodo y que los flujos
        respeten las capacidades mínimas y máximas de los arcos.
        
        Returns:
            float: Valor de calidad de la solución (valor del flujo si es factible,
                  infinito en caso contrario)
        """
        # Verificar conservación de flujo en cada nodo
        self.factible = True
        for nodo in self.grafo.nodes():
            if nodo == self.origen or nodo == self.destino:
                continue
            
            flujo_entrante = sum(self.flujos.get((i, nodo), 0) for i in self.grafo.predecessors(nodo))
            flujo_saliente = sum(self.flujos.get((nodo, j), 0) for j in self.grafo.successors(nodo))
            
            if abs(flujo_entrante - flujo_saliente) > 1e-6:
                self.factible = False
        
        # Verificar capacidades
        for u, v, data in self.grafo.edges(data=True):
            if self.flujos.get((u, v), 0) < data['lower'] or self.flujos.get((u, v), 0) > data['capacity']:
                self.factible = False
        
        # Calcular valor del flujo
        flujo_saliente_origen = sum(self.flujos.get((self.origen, j), 0) for j in self.grafo.successors(self.origen))
        flujo_entrante_origen = sum(self.flujos.get((i, self.origen), 0) for i in self.grafo.predecessors(self.origen))
        self.valor_flujo = flujo_saliente_origen - flujo_entrante_origen
        
        # Calcular calidad (objetivo es minimizar)
        self.calidad = self.valor_flujo if self.factible else float('inf')
        return self.calidad
    
    def copiar(self):
        """
        Crea una copia profunda de la solución actual.
        
        Returns:
            SolucionMinFlow: Una nueva instancia con los mismos valores
                            pero independiente de la original
        """
        nueva_solucion = SolucionMinFlow(self.flujos.copy(), self.grafo, self.origen, self.destino)
        nueva_solucion.valor_flujo = self.valor_flujo
        nueva_solucion.factible = self.factible
        nueva_solucion.calidad = self.calidad
        return nueva_solucion
    
    def visualizar_solucion(self, titulo="Solución de Flujo Mínimo"):
        """
        Visualiza la solución como un grafo con los flujos en cada arco.
        
        Args:
            titulo (str): Título para la visualización
        """
        plt.figure(figsize=(14, 10))
        
        # Usar un layout más estable para visualización consistente
        pos = nx.spring_layout(self.grafo, seed=42, k=1, iterations=50)
        
        # Dibujar nodos con colores según su tipo
        nx.draw_networkx_nodes(self.grafo, pos, node_color='lightblue', 
                              node_size=700, alpha=0.8, label='Nodos intermedios')
        
        # Resaltar origen y destino con colores distintivos
        nx.draw_networkx_nodes(self.grafo, pos, nodelist=[self.origen], 
                              node_color='green', node_size=800, alpha=0.8, label='Origen')
        nx.draw_networkx_nodes(self.grafo, pos, nodelist=[self.destino], 
                              node_color='red', node_size=800, alpha=0.8, label='Destino')
        
        # Dibujar etiquetas de nodos con mayor tamaño
        nx.draw_networkx_labels(self.grafo, pos, font_weight='bold', font_size=14)
        
        # Dibujar arcos con etiquetas de flujo mejoradas
        edge_labels = {}
        for u, v, data in self.grafo.edges(data=True):
            flujo = self.flujos.get((u, v), 0)
            edge_labels[(u, v)] = f"{flujo:.2f}/{data['capacity']}"
        
        # Colorear arcos según su flujo relativo a la capacidad
        edge_colors = []
        widths = []
        for u, v, data in self.grafo.edges(data=True):
            flujo = self.flujos.get((u, v), 0)
            ratio = flujo / data['capacity']
            
            # Usar un mapa de colores más intuitivo (verde a rojo)
            if ratio < 0.3:
                color = (0.2, 0.8, 0.2, 0.7)  # Verde para flujos bajos
            elif ratio < 0.7:
                color = (0.8, 0.8, 0.2, 0.7)  # Amarillo para flujos medios
            else:
                color = (0.8, 0.2, 0.2, 0.7)  # Rojo para flujos altos
                
            edge_colors.append(color)
            widths.append(1 + 4 * ratio)  # Ancho proporcional al flujo
        
        # Dibujar arcos con flechas más grandes
        nx.draw_networkx_edges(self.grafo, pos, width=widths, edge_color=edge_colors, 
                              arrowsize=25, alpha=0.8, connectionstyle='arc3,rad=0.1')
        
        # Dibujar etiquetas de arcos con mejor formato
        nx.draw_networkx_edge_labels(self.grafo, pos, edge_labels=edge_labels, 
                                    font_size=12, font_weight='bold')
        
        # Añadir título y leyenda
        plt.title(titulo, fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        
        # Añadir información sobre el valor del flujo
        plt.figtext(0.5, 0.01, f"Valor del flujo: {self.valor_flujo:.2f}", 
                   ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Guardar la figura con alta resolución
        plt.savefig(f"{titulo.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

class ScatterSearchMinFlow:
    """
    Implementación del algoritmo Scatter Search para resolver el problema de flujo mínimo.
    
    El algoritmo combina diversificación, intensificación y mejora local para
    encontrar soluciones de alta calidad al problema de flujo mínimo en una red.
    """
    def __init__(self, grafo, origen, destino, tam_refset=10, max_iter=100):
        """
        Inicializa el algoritmo Scatter Search.
        
        Args:
            grafo (DiGraph): Grafo dirigido que representa la red
            origen (node): Nodo origen del flujo
            destino (node): Nodo destino del flujo
            tam_refset (int): Tamaño del conjunto de referencia
            max_iter (int): Número máximo de iteraciones
        """
        self.grafo = grafo
        self.origen = origen
        self.destino = destino
        self.tam_refset = tam_refset
        self.max_iter = max_iter
        self.refset = []  # Conjunto de referencia
        self.mejor_solucion = None
        self.historial_mejor = []
        self.solucion_exacta = None  # Para almacenar la solución del modelo exacto
    
    def generar_solucion_inicial_relajada(self):
        """
        Genera una solución inicial usando una relajación lineal del problema.
        
        Utiliza programación lineal para encontrar una solución inicial de buena
        calidad que puede servir como punto de partida para el algoritmo.
        
        Returns:
            SolucionMinFlow: Una solución inicial basada en la relajación lineal
        """
        # Crear modelo relajado (sin restricciones de integralidad)
        arcos = list(self.grafo.edges())
        n_arcos = len(arcos)
        
        # Mapeo de arcos a índices
        arco_a_indice = {arco: i for i, arco in enumerate(arcos)}
        
        # Coeficientes de la función objetivo (todos 1 para minimizar flujo total)
        c = np.ones(n_arcos)
        
        # Restricciones de conservación de flujo
        A_eq = []
        b_eq = []
        
        for nodo in self.grafo.nodes():
            if nodo == self.origen:
                b_eq.append(1)  # Flujo neto saliente = 1 (valor arbitrario)
            elif nodo == self.destino:
                b_eq.append(-1)  # Flujo neto entrante = -1
            else:
                b_eq.append(0)  # Conservación de flujo
            
            fila = np.zeros(n_arcos)
            
            # Arcos salientes (flujo positivo)
            for sucesor in self.grafo.successors(nodo):
                if (nodo, sucesor) in arco_a_indice:
                    fila[arco_a_indice[(nodo, sucesor)]] = 1
            
            # Arcos entrantes (flujo negativo)
            for predecesor in self.grafo.predecessors(nodo):
                if (predecesor, nodo) in arco_a_indice:
                    fila[arco_a_indice[(predecesor, nodo)]] = -1
            
            A_eq.append(fila)
        
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Límites de capacidad
        bounds = []
        for u, v, data in self.grafo.edges(data=True):
            bounds.append((data['lower'], data['capacity']))
        
        # Resolver el problema relajado
        resultado = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if resultado.success:
            # Crear solución a partir del resultado
            flujos = {}
            for i, arco in enumerate(arcos):
                flujos[arco] = resultado.x[i]
            
            solucion = SolucionMinFlow(flujos, self.grafo, self.origen, self.destino)
            solucion.evaluar()
            
            # Guardar como solución exacta
            self.solucion_exacta = solucion
            
            return solucion
        else:
            # Si falla, crear una solución aleatoria
            return self.generar_solucion_aleatoria()
    
    def generar_solucion_aleatoria(self):
        """
        Genera una solución aleatoria para el problema.
        
        Asigna valores aleatorios a los flujos en cada arco dentro de sus límites
        y luego repara la solución para hacerla factible.
        
        Returns:
            SolucionMinFlow: Una solución aleatoria factible
        """
        flujos = {}
        for u, v, data in self.grafo.edges(data=True):
            flujos[(u, v)] = random.uniform(data['lower'], data['capacity'])
        
        # Aplicar reparación para hacerla factible
        solucion = SolucionMinFlow(flujos, self.grafo, self.origen, self.destino)
        return self.reparar_solucion(solucion)
    
    def reparar_solucion(self, solucion):
        """
        Repara una solución para hacerla factible.
        
        Ajusta los flujos para garantizar la conservación de flujo en cada nodo
        y que se respeten las capacidades de los arcos.
        
        Args:
            solucion (SolucionMinFlow): Solución a reparar
            
        Returns:
            SolucionMinFlow: Solución reparada y factible
        """
        # Crear una copia para no modificar la original
        nueva_solucion = solucion.copiar()
        
        # Reparar conservación de flujo en cada nodo
        for nodo in self.grafo.nodes():
            if nodo == self.origen or nodo == self.destino:
                continue
            
            flujo_entrante = sum(nueva_solucion.flujos.get((i, nodo), 0) for i in self.grafo.predecessors(nodo))
            flujo_saliente = sum(nueva_solucion.flujos.get((nodo, j), 0) for j in self.grafo.successors(nodo))
            
            # Si hay desequilibrio, ajustar flujos salientes
            if abs(flujo_entrante - flujo_saliente) > 1e-6:
                sucesores = list(self.grafo.successors(nodo))
                if sucesores:
                    # Distribuir la diferencia entre los arcos salientes
                    diferencia = flujo_entrante - flujo_saliente
                    arcos_ajustables = []
                    
                    for j in sucesores:
                        arco = (nodo, j)
                        capacidad = self.grafo[nodo][j]['capacity']
                        limite_inferior = self.grafo[nodo][j]['lower']
                        flujo_actual = nueva_solucion.flujos.get(arco, 0)
                        
                        if diferencia > 0 and flujo_actual < capacidad:
                            arcos_ajustables.append((arco, min(diferencia, capacidad - flujo_actual)))
                        elif diferencia < 0 and flujo_actual > limite_inferior:
                            arcos_ajustables.append((arco, max(diferencia, limite_inferior - flujo_actual)))
                    
                    if arcos_ajustables:
                        # Seleccionar un arco al azar para ajustar
                        arco_elegido, ajuste_max = random.choice(arcos_ajustables)
                        nueva_solucion.flujos[arco_elegido] = nueva_solucion.flujos.get(arco_elegido, 0) + ajuste_max
        
        # Verificar capacidades y ajustar si es necesario
        for u, v, data in self.grafo.edges(data=True):
            arco = (u, v)
            if arco not in nueva_solucion.flujos:
                nueva_solucion.flujos[arco] = data['lower']
            elif nueva_solucion.flujos[arco] < data['lower']:
                nueva_solucion.flujos[arco] = data['lower']
            elif nueva_solucion.flujos[arco] > data['capacity']:
                nueva_solucion.flujos[arco] = data['capacity']
        
        nueva_solucion.evaluar()
        return nueva_solucion
    
    def diversificacion(self, num_soluciones=100):
        """
        Genera un conjunto diverso de soluciones iniciales.
        
        Combina una solución basada en relajación lineal con soluciones
        aleatorias para crear un conjunto inicial diverso.
        
        Args:
            num_soluciones (int): Número de soluciones a generar
            
        Returns:
            list: Lista de soluciones iniciales diversas
        """
        soluciones = []
        
        # Generar solución usando relajación
        sol_relajada = self.generar_solucion_inicial_relajada()
        soluciones.append(sol_relajada)
        
        # Generar soluciones aleatorias
        for _ in range(num_soluciones - 1):
            soluciones.append(self.generar_solucion_aleatoria())
        
        return soluciones
    
    def actualizar_refset(self, soluciones):
        """
        Actualiza el conjunto de referencia con las mejores soluciones.
        
        Selecciona las mejores soluciones por calidad y las más diversas
        para mantener un balance entre intensificación y diversificación.
        
        Args:
            soluciones (list): Lista de soluciones candidatas
        """
        # Ordenar por calidad
        soluciones_ordenadas = sorted(soluciones, key=lambda x: x.calidad)
        
        # Seleccionar las mejores
        mejores = soluciones_ordenadas[:self.tam_refset // 2]
        
        # Seleccionar las más diversas del resto
        resto = soluciones_ordenadas[self.tam_refset // 2:]
        diversas = []
        
        if resto:
            # Medida de diversidad: distancia euclidiana entre vectores de flujo
            for _ in range(self.tam_refset - len(mejores)):
                if not resto:
                    break
                
                # Encontrar la solución más diversa respecto a las ya seleccionadas
                max_distancia = -1
                sol_mas_diversa = None
                idx_mas_diversa = -1
                
                for i, sol in enumerate(resto):
                    # Calcular distancia mínima a las soluciones ya seleccionadas
                    min_distancia = float('inf')
                    
                    for sol_sel in mejores + diversas:
                        # Calcular distancia euclidiana entre flujos
                        distancia = 0
                        for arco in set(sol.flujos.keys()).union(sol_sel.flujos.keys()):
                            val1 = sol.flujos.get(arco, 0)
                            val2 = sol_sel.flujos.get(arco, 0)
                            distancia += (val1 - val2) ** 2
                        distancia = np.sqrt(distancia)
                        
                        min_distancia = min(min_distancia, distancia)
                    
                    if min_distancia > max_distancia:
                        max_distancia = min_distancia
                        sol_mas_diversa = sol
                        idx_mas_diversa = i
                
                if sol_mas_diversa:
                    diversas.append(sol_mas_diversa)
                    resto.pop(idx_mas_diversa)
        
        self.refset = mejores + diversas
    
    def combinar_soluciones(self):
        """
        Combina pares de soluciones del conjunto de referencia.
        
        Utiliza dos métodos de combinación:
        1. Promedio ponderado de flujos
        2. Cruce de arcos (selección aleatoria de valores)
        
        Returns:
            list: Lista de nuevas soluciones generadas por combinación
        """
        nuevas_soluciones = []
        
        for i in range(len(self.refset)):
            for j in range(i+1, len(self.refset)):
                sol1 = self.refset[i]
                sol2 = self.refset[j]
                
                # Combinación por promedio ponderado
                flujos_combinados = {}
                for arco in set(sol1.flujos.keys()).union(sol2.flujos.keys()):
                    val1 = sol1.flujos.get(arco, 0)
                    val2 = sol2.flujos.get(arco, 0)
                    peso = random.uniform(0, 1)
                    flujos_combinados[arco] = val1 * peso + val2 * (1 - peso)
                
                nueva_sol = SolucionMinFlow(flujos_combinados, self.grafo, self.origen, self.destino)
                nueva_sol = self.reparar_solucion(nueva_sol)
                nuevas_soluciones.append(nueva_sol)
                
                # Combinación por cruce de arcos
                flujos_cruzados = {}
                for arco in set(sol1.flujos.keys()).union(sol2.flujos.keys()):
                    if random.random() < 0.5:
                        flujos_cruzados[arco] = sol1.flujos.get(arco, 0)
                    else:
                        flujos_cruzados[arco] = sol2.flujos.get(arco, 0)
                
                nueva_sol = SolucionMinFlow(flujos_cruzados, self.grafo, self.origen, self.destino)
                nueva_sol = self.reparar_solucion(nueva_sol)
                nuevas_soluciones.append(nueva_sol)
        
        return nuevas_soluciones
    
    def mejora_local(self, solucion):
        """
        Aplica una búsqueda local para mejorar la solución.
        
        Intenta reducir el flujo en algunos arcos para minimizar
        el valor total del flujo manteniendo la factibilidad.
        
        Args:
            solucion (SolucionMinFlow): Solución a mejorar
            
        Returns:
            SolucionMinFlow: Solución mejorada
        """
        mejor_sol = solucion.copiar()
        mejor_calidad = mejor_sol.calidad
        
        # Intentar reducir el flujo en algunos arcos
        for arco in mejor_sol.flujos:
            flujo_original = mejor_sol.flujos[arco]
            
            # Intentar reducir el flujo
            for reduccion in [0.1, 0.05, 0.01]:
                nueva_sol = mejor_sol.copiar()
                nueva_sol.flujos[arco] = max(self.grafo[arco[0]][arco[1]]['lower'], 
                                            flujo_original - reduccion * flujo_original)
                
                # Reparar para mantener factibilidad
                nueva_sol = self.reparar_solucion(nueva_sol)
                
                if nueva_sol.factible and nueva_sol.calidad < mejor_calidad:
                    mejor_sol = nueva_sol
                    mejor_calidad = nueva_sol.calidad
        
        return mejor_sol
    
    def ejecutar(self):
        """
        Ejecuta el algoritmo Scatter Search completo.
        
        Implementa las fases de diversificación, combinación, mejora y
        actualización del conjunto de referencia durante varias iteraciones.
        
        Returns:
            SolucionMinFlow: La mejor solución encontrada
        """
        print("Iniciando Scatter Search para el problema de Minimum Flow...")
        
        # Fase de diversificación
        print("Fase de diversificación...")
        soluciones_iniciales = self.diversificacion()
        
        # Inicializar conjunto de referencia
        self.actualizar_refset(soluciones_iniciales)
        
        # Guardar la mejor solución inicial
        self.mejor_solucion = min(self.refset, key=lambda x: x.calidad)
        self.historial_mejor.append(self.mejor_solucion.calidad)
        
        print(f"Mejor solución inicial: {self.mejor_solucion.calidad}")
        
        # Iteraciones principales
        for iter in range(self.max_iter):
            print(f"Iteración {iter+1}/{self.max_iter}")
            
            # Fase de combinación
            nuevas_soluciones = self.combinar_soluciones()
            
            # Fase de mejora
            for i in range(len(nuevas_soluciones)):
                nuevas_soluciones[i] = self.mejora_local(nuevas_soluciones[i])
            
            # Actualizar conjunto de referencia
            self.actualizar_refset(self.refset + nuevas_soluciones)
            
            # Actualizar mejor solución
            mejor_actual = min(self.refset, key=lambda x: x.calidad)
            if mejor_actual.calidad < self.mejor_solucion.calidad:
                self.mejor_solucion = mejor_actual
                print(f"  Nueva mejor solución encontrada: {self.mejor_solucion.calidad}")
            
            self.historial_mejor.append(self.mejor_solucion.calidad)
        
        print("\nBúsqueda finalizada.")
        print(f"Mejor valor de flujo encontrado: {self.mejor_solucion.calidad}")
        
        # Graficar convergencia
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historial_mejor)), self.historial_mejor, 'b-', 
                linewidth=2, label='Valor del flujo')
        
        # Si hay solución exacta, añadir línea horizontal
        if self.solucion_exacta:
            plt.axhline(y=self.solucion_exacta.calidad, color='r', linestyle='--', 
                       label=f'Valor óptimo: {self.solucion_exacta.calidad:.4f}')
            
            # Calcular gap porcentual
            gap = ((self.solucion_exacta.calidad - self.mejor_solucion.calidad) / 
                   self.solucion_exacta.calidad) * 100
            plt.text(len(self.historial_mejor) * 0.7, self.mejor_solucion.calidad, 
                     f'Gap: {gap:.2f}%', fontsize=10, verticalalignment='bottom')
        
        plt.title('Convergencia de Scatter Search para Minimum Flow', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Valor del flujo mínimo', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir leyenda solo si hay elementos para mostrar
        if self.solucion_exacta:
            plt.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('convergencia_scatter_search.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Visualizar soluciones
        if self.solucion_exacta:
            print("\nComparando soluciones:")
            print(f"Valor del flujo (modelo exacto): {self.solucion_exacta.calidad}")
            print(f"Valor del flujo (metaheurística): {self.mejor_solucion.calidad}")
            print(f"Diferencia: {abs(self.solucion_exacta.calidad - self.mejor_solucion.calidad)}")
            print(f"Gap porcentual: {gap:.2f}%")
            
            # Visualizar solución exacta
            self.solucion_exacta.visualizar_solucion("Solución del Modelo Exacto")
            
            # Visualizar solución de la metaheurística
            self.mejor_solucion.visualizar_solucion("Solución de la Metaheurística Scatter Search")
            
            # Visualizar comparación lado a lado
            self.visualizar_comparacion_soluciones()
        
        return self.mejor_solucion
        
    def visualizar_comparacion_soluciones(self):
        """
        Visualiza una comparación lado a lado de la solución óptima y la encontrada
        por el algoritmo Scatter Search.
        """
        if not self.solucion_exacta or not self.mejor_solucion:
            print("No hay soluciones para comparar")
            return
            
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Configuración común para ambos subplots
        pos = nx.spring_layout(self.grafo, seed=42, k=1, iterations=50)
        
        # Función auxiliar para dibujar un grafo en un subplot
        def dibujar_grafo(ax, solucion, titulo):
            # Dibujar nodos
            nx.draw_networkx_nodes(self.grafo, pos, node_color='lightblue', 
                                  node_size=700, alpha=0.8, ax=ax)
            
            # Resaltar origen y destino
            nx.draw_networkx_nodes(self.grafo, pos, nodelist=[self.origen], 
                                  node_color='green', node_size=800, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(self.grafo, pos, nodelist=[self.destino], 
                                  node_color='red', node_size=800, alpha=0.8, ax=ax)
            
            # Dibujar etiquetas de nodos
            nx.draw_networkx_labels(self.grafo, pos, font_weight='bold', font_size=14, ax=ax)
            
            # Dibujar arcos con etiquetas de flujo
            edge_labels = {}
            for u, v, data in self.grafo.edges(data=True):
                flujo = solucion.flujos.get((u, v), 0)
                edge_labels[(u, v)] = f"{flujo:.2f}/{data['capacity']}"
            
            # Colorear arcos según su flujo relativo a la capacidad
            edge_colors = []
            widths = []
            for u, v, data in self.grafo.edges(data=True):
                flujo = solucion.flujos.get((u, v), 0)
                ratio = flujo / data['capacity']
                
                # Usar un mapa de colores más intuitivo (verde a rojo)
                if ratio < 0.3:
                    color = (0.2, 0.8, 0.2, 0.7)  # Verde para flujos bajos
                elif ratio < 0.7:
                    color = (0.8, 0.8, 0.2, 0.7)  # Amarillo para flujos medios
                else:
                    color = (0.8, 0.2, 0.2, 0.7)  # Rojo para flujos altos
                    
                edge_colors.append(color)
                widths.append(1 + 4 * ratio)  # Ancho proporcional al flujo
            
            # Dibujar arcos con flechas
            nx.draw_networkx_edges(self.grafo, pos, width=widths, edge_color=edge_colors, 
                                  arrowsize=25, alpha=0.8, connectionstyle='arc3,rad=0.1', ax=ax)
            
            # Dibujar etiquetas de arcos
            nx.draw_networkx_edge_labels(self.grafo, pos, edge_labels=edge_labels, 
                                        font_size=12, font_weight='bold', ax=ax)
            
            # Añadir título
            ax.set_title(titulo, fontsize=16, fontweight='bold')
            
            # Añadir información sobre el valor del flujo
            ax.text(0.5, 0.01, f"Valor del flujo: {solucion.valor_flujo:.2f}", 
                   ha='center', fontsize=14, transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
            
            ax.axis('off')
        
        # Dibujar solución óptima
        dibujar_grafo(ax1, self.solucion_exacta, "Solución Óptima")
        
        # Dibujar solución encontrada
        dibujar_grafo(ax2, self.mejor_solucion, "Solución Scatter Search")
        
        # Añadir título general
        gap = ((self.solucion_exacta.calidad - self.mejor_solucion.calidad) / 
               self.solucion_exacta.calidad) * 100
        fig.suptitle(f"Comparación de Soluciones - Gap: {gap:.2f}%", 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("comparacion_soluciones.png", dpi=300, bbox_inches='tight')
        plt.show()

# Ejemplo de uso
def crear_grafo_ejemplo():
    """
    Crea un grafo de ejemplo para el problema de flujo mínimo.
    
    Construye una red con 5 nodos y 7 arcos con capacidades y límites
    inferiores para demostrar el funcionamiento del algoritmo.
    
    Returns:
        tuple: (grafo, origen, destino) donde grafo es un DiGraph de NetworkX
    """
    G = nx.DiGraph()
    
    # Añadir nodos
    G.add_node('s')  # origen
    G.add_node('t')  # destino
    G.add_node('a')
    G.add_node('b')
    G.add_node('c')
    
    # Añadir arcos con capacidades y límites inferiores
    # (origen, destino, {capacidad, límite inferior})
    G.add_edge('s', 'a', capacity=10, lower=1)
    G.add_edge('s', 'b', capacity=8, lower=0)
    G.add_edge('a', 'b', capacity=5, lower=0)
    G.add_edge('a', 'c', capacity=7, lower=2)
    G.add_edge('b', 'c', capacity=6, lower=0)
    G.add_edge('b', 't', capacity=8, lower=1)
    G.add_edge('c', 't', capacity=10, lower=3)
    
    return G, 's', 't'

if __name__ == "__main__":
    # Crear grafo de ejemplo
    grafo, origen, destino = crear_grafo_ejemplo()
    
    # Ejecutar Scatter Search
    ss = ScatterSearchMinFlow(grafo, origen, destino, tam_refset=10, max_iter=50)
    mejor_solucion = ss.ejecutar()
    
    # Mostrar detalles de la solución
    print("\nDetalles de la mejor solución encontrada:")
    print(f"Factible: {mejor_solucion.factible}")
    print(f"Valor del flujo: {mejor_solucion.valor_flujo}")
    print("\nFlujos en los arcos:")
    for arco, flujo in sorted(mejor_solucion.flujos.items()):
        print(f"  {arco}: {flujo:.4f}")
    
    # Forzar la visualización de los gráficos
    plt.show(block=True)
    
    # Pequeña pausa para asegurar que los gráficos se muestren
    time.sleep(1)
    
    # Visualizar la solución final
    mejor_solucion.visualizar_solucion("Solución Final")
    plt.show(block=True)
    
    # Mantener las ventanas abiertas
    input("\nPresiona Enter para cerrar las ventanas de gráficos...")
    plt.close('all')
