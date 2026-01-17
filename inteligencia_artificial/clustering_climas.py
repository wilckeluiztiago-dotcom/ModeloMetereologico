import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: CLUSTERING CLIMÁTICO (K-MEANS)
=================================================================

Algoritmo não-supervisionado para agrupar estações ou dias em "Tipos Climáticos".
Implementação manual do algoritmo K-Means (Lloyd's algorithm).

Aplicações:
- Zoneamento de microrregiões (Cluster 1: Serra, Cluster 2: Litoral...)
- Identificação de padrões diários (Dia Tipo 1: Frio/Seco, Dia Tipo 2: Quente/Úmido)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class KMeansClimatico:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroides = None
        self.labels = None
        
    def fit(self, X):
        """
        X: (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 1. Inicialização Aleatória (Forgy method)
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroides = X[indices]
        
        for i in range(self.max_iter):
            # 2. Assignment Step
            # Calcular distância de cada ponto para cada centroide
            # Distância Euclidiana
            distancias = np.zeros((n_samples, self.k))
            for c in range(self.k):
                # Broadcasting
                dist = np.linalg.norm(X - self.centroides[c], axis=1)
                distancias[:, c] = dist
            
            novos_labels = np.argmin(distancias, axis=1)
            
            # Checar convergência
            if i > 0 and np.all(novos_labels == self.labels):
                break
                
            self.labels = novos_labels
            
            # 3. Update Step
            novos_centroides = np.zeros((self.k, n_features))
            for c in range(self.k):
                mask = self.labels == c
                if np.any(mask):
                    novos_centroides[c] = np.mean(X[mask], axis=0)
                else:
                    # Se cluster vazio, reinicializa aleatoriamente
                    novos_centroides[c] = X[np.random.choice(n_samples)]
            
            self.centroides = novos_centroides
            
        return self.labels, self.centroides

    def inercia(self, X):
        """Soma dos quadrados das distâncias intra-cluster (WCSS)."""
        wcss = 0
        distancias = np.linalg.norm(X - self.centroides[self.labels], axis=1)
        wcss = np.sum(distancias**2)
        return wcss

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando K-Means Climático...")
    
    # Criar 3 clusters sintéticos 2D (Temp vs Umidade)
    c1 = np.random.normal([10, 80], 2, (50, 2)) # Frio/Úmido
    c2 = np.random.normal([30, 40], 3, (50, 2)) # Quente/Seco
    c3 = np.random.normal([20, 60], 4, (50, 2)) # Ameno
    
    X = np.vstack([c1, c2, c3])
    
    kmeans = KMeansClimatico(k=3)
    labels, centers = kmeans.fit(X)
    
    print(f"Centroides Encontrados:\n{centers}")
    print(f"Inércia Final: {kmeans.inercia(X):.2f}")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, label='Centroides')
    plt.xlabel('Temperatura')
    plt.ylabel('Umidade')
    plt.title('Zoneamento Climático Automático (K-Means)')
    plt.legend()
    # plt.show()
    print("Gráfico Clustering gerado.")
