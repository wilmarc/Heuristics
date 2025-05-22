import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Clase para la partícula del PSO
class Particula:
    def __init__(self, dim):
        # Inicializar posición y velocidad aleatoriamente
        self.posicion = np.random.uniform(-1, 1, dim)
        self.velocidad = np.random.uniform(-0.1, 0.1, dim)
        self.mejor_posicion = self.posicion.copy()
        self.fitness = float('inf')
        self.mejor_fitness = float('inf')
    
    def actualizar_velocidad(self, mejor_global, w, c1, c2):
        r1 = np.random.random(len(self.posicion))
        r2 = np.random.random(len(self.posicion))
        
        componente_inercia = w * self.velocidad
        componente_cognitivo = c1 * r1 * (self.mejor_posicion - self.posicion)
        componente_social = c2 * r2 * (mejor_global - self.posicion)
        
        self.velocidad = componente_inercia + componente_cognitivo + componente_social
    
    def actualizar_posicion(self):
        self.posicion = self.posicion + self.velocidad
        
    def evaluar_fitness(self, X, y):
        # Calcular el error cuadrático medio como función de fitness
        y_pred = X @ self.posicion
        mse = np.mean((y - y_pred) ** 2)
        self.fitness = mse
        
        # Actualizar mejor posición personal si mejora
        if self.fitness < self.mejor_fitness:
            self.mejor_fitness = self.fitness
            self.mejor_posicion = self.posicion.copy()
        
        return self.fitness

# Clase para el optimizador PSO para regresión lineal
class PSO_Regresion:
    def __init__(self, X, y, tam_poblacion=50, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        self.X = X
        self.y = y
        self.dim = X.shape[1]  # Dimensión = número de características
        self.tam_poblacion = tam_poblacion
        self.max_iter = max_iter
        self.w = w  # Factor de inercia
        self.c1 = c1  # Factor cognitivo
        self.c2 = c2  # Factor social
        
        # Inicializar partículas
        self.particulas = [Particula(self.dim) for _ in range(tam_poblacion)]
        
        # Inicializar mejor global
        self.mejor_global_posicion = np.zeros(self.dim)
        self.mejor_global_fitness = float('inf')
        
        # Historial para análisis
        self.historial_fitness = []
    
    def ejecutar(self):
        # Evaluar fitness inicial de todas las partículas
        for p in self.particulas:
            fitness = p.evaluar_fitness(self.X, self.y)
            if fitness < self.mejor_global_fitness:
                self.mejor_global_fitness = fitness
                self.mejor_global_posicion = p.posicion.copy()
        
        # Iterar el algoritmo
        for i in range(self.max_iter):
            # Actualizar posiciones
            for p in self.particulas:
                p.actualizar_posicion()
                fitness = p.evaluar_fitness(self.X, self.y)
                
                # Actualizar mejor global si es necesario
                if fitness < self.mejor_global_fitness:
                    self.mejor_global_fitness = fitness
                    self.mejor_global_posicion = p.posicion.copy()
            
            # Actualizar velocidades
            for p in self.particulas:
                p.actualizar_velocidad(self.mejor_global_posicion, self.w, self.c1, self.c2)
            
            # Guardar historial
            self.historial_fitness.append(self.mejor_global_fitness)
            
            # Imprimir progreso
            if (i+1) % 10 == 0:
                print(f"Iteración {i+1}/{self.max_iter}, Mejor MSE: {self.mejor_global_fitness:.6f}")
    
    def predecir(self, X):
        return X @ self.mejor_global_posicion
    
    def mostrar_resultados(self):
        print("\nResultados finales:")
        print(f"Mejor MSE encontrado: {self.mejor_global_fitness:.6f}")
        print(f"Coeficientes: {self.mejor_global_posicion}")
        
        # Graficar convergencia
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historial_fitness)), self.historial_fitness, 'g-')
        plt.title('Convergencia del algoritmo PSO')
        plt.xlabel('Iteración')
        plt.ylabel('Error Cuadrático Medio (MSE)')
        plt.grid(True)
        plt.show()

# Cargar el dataset de California Housing
print("Cargando dataset de California Housing...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Añadir término de sesgo (intercept)
X_train_scaled = np.column_stack((np.ones(X_train_scaled.shape[0]), X_train_scaled))
X_test_scaled = np.column_stack((np.ones(X_test_scaled.shape[0]), X_test_scaled))

print(f"Dimensiones de los datos: X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")

# Solución con PSO
print("\n--- Solución con PSO ---")
pso = PSO_Regresion(
    X=X_train_scaled,
    y=y_train,
    tam_poblacion=50,
    max_iter=100,
    w=0.7,
    c1=1.5,
    c2=1.5
)

pso.ejecutar()
pso.mostrar_resultados()

# Evaluar en conjunto de prueba
y_pred_pso = pso.predecir(X_test_scaled)
mse_pso = mean_squared_error(y_test, y_pred_pso)
r2_pso = r2_score(y_test, y_pred_pso)

print(f"\nEvaluación en conjunto de prueba (PSO):")
print(f"MSE: {mse_pso:.6f}")
print(f"R²: {r2_pso:.6f}")

# Solución con OLS (Ordinary Least Squares)
print("\n--- Solución con OLS ---")
ols = LinearRegression(fit_intercept=False)  # False porque ya añadimos la columna de unos
ols.fit(X_train_scaled, y_train)

# Evaluar en conjunto de prueba
y_pred_ols = ols.predict(X_test_scaled)
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

print(f"Coeficientes OLS: {ols.coef_}")
print(f"\nEvaluación en conjunto de prueba (OLS):")
print(f"MSE: {mse_ols:.6f}")
print(f"R²: {r2_ols:.6f}")


#Solución con Randomforest

model_RF=RandomForestRegressor(n_estimators=500, random_state=0)
model_RF.fit(X_train_scaled, y_train)

#Evaluación en el conjunto de prueba
y_pred_RF=model_RF.predict(X_test_scaled)
mse_RF = mean_squared_error(y_test, y_pred_RF)
r2_RF = r2_score(y_test, y_pred_RF)

print(f"Coeficientes Random Forest: {model_RF.get_params()}")
print(f"\nEvaluación en conjunto de prueba (RF):")
print(f"MSE: {mse_RF:.6f}")
print(f"R²: {r2_RF:.6f}")


# Comparar resultados
print("\n--- Comparación de resultados ---")
print(f"MSE - PSO: {mse_pso:.3f}, OLS: {mse_ols:.3f}, RF: {mse_RF:.3f}")
print(f"R² - PSO: {r2_pso:.3f}, OLS: {r2_ols:.3f}, RF {r2_RF:.3f}")


# Visualizar predicciones
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_pso, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones PSO')
plt.title('PSO: Valores reales vs. predicciones')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ols, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones OLS')
plt.title('OLS: Valores reales vs. predicciones')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_RF, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones RF')
plt.title('RF: Valores reales vs. predicciones')

plt.tight_layout()
plt.show()
