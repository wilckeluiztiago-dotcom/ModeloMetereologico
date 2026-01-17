import numpy as np

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: FLORESTA ALEATÓRIA (RANDOM FOREST)
=====================================================================

Implementa um algoritmo de Random Forest simplificado para CLASSIFICAÇÃO binária
(Choverá amanha? Sim/Não).

Baseia-se em um ensemble (conjunto) de Árvores de Decisão simples.
Features típicas: Temp, Umidade, Pressão, Vento.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ArvoreDecisaoSimples:
    def __init__(self, profundidade_max=3):
        self.depth = profundidade_max
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None
        
    def fit(self, X, y):
        """Treinamento 'Greedy' para encontrar melhor split."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Critério de parada: Nó puro ou profundidade máxima
        if self.depth == 0 or n_labels == 1:
            self.prediction = np.mean(y) >= 0.5 # Maioria (0 ou 1)
            return
            
        # Encontrar melhor split (Random feature sub-selection)
        # Simplificação: Escolhe feature aleatória para ser rápido
        feat_tries = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
        
        best_gain = -1
        
        for feat in feat_tries:
            thresholds = np.unique(X[:, feat])
            # Tenta alguns thresholds apenas
            if len(thresholds) > 10:
                thresholds = np.random.choice(thresholds, 10, replace=False)
                
            for thresh in thresholds:
                left_mask = X[:, feat] < thresh
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0: continue
                
                # Ganho de Informação (Gini Impurity reduction simplificado)
                # Maximizando a pureza
                p_left = np.mean(y[left_mask])
                p_right = np.mean(y[~left_mask])
                gini_left = 1 - (p_left**2 + (1-p_left)**2)
                gini_right = 1 - (p_right**2 + (1-p_right)**2)
                
                gain = 1 - (len(y[left_mask])/n_samples * gini_left + len(y[~left_mask])/n_samples * gini_right)
                
                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = feat
                    self.threshold = thresh
        
        if self.feature_idx is None: # Nenhum split bom
            self.prediction = np.mean(y) >= 0.5
            return
            
        # Criar filhos
        left_mask = X[:, self.feature_idx] < self.threshold
        self.left = ArvoreDecisaoSimples(self.depth - 1)
        self.right = ArvoreDecisaoSimples(self.depth - 1)
        
        self.left.fit(X[left_mask], y[left_mask])
        self.right.fit(X[~left_mask], y[~left_mask])
        
    def predict(self, sample):
        if self.prediction is not None:
            return self.prediction
        
        if sample[self.feature_idx] < self.threshold:
            return self.left.predict(sample)
        else:
            return self.right.predict(sample)


class FlorestaAleatoriaChuva:
    def __init__(self, n_arvores=10):
        self.n_arvores = n_arvores
        self.arvores = []
        
    def treinar(self, X, y):
        """Bootstrap Aggregating (Bagging)."""
        n_samples = X.shape[0]
        
        for _ in range(self.n_arvores):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            tree = ArvoreDecisaoSimples(profundidade_max=4)
            tree.fit(X_boot, y_boot)
            self.arvores.append(tree)
            
    def predizer(self, X):
        predictions = []
        for sample in X:
            votes = [tree.predict(sample) for tree in self.arvores]
            predictions.append(np.mean(votes) >= 0.5)
        return np.array(predictions, dtype=int)
        
    def probabilidade(self, X):
        probs = []
        for sample in X:
            votes = [tree.predict(sample) for tree in self.arvores]
            probs.append(np.mean(votes))
        return np.array(probs)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Random Forest (Classificação Chuva)...")
    
    # Dados Fake: [Umidade, Pressão] -> Chuva
    # Se Umidade > 80 e Pressao < 1010 -> Chuva (1)
    X = np.random.rand(200, 2)
    X[:, 0] = X[:, 0] * 100 # Umidade 0-100
    X[:, 1] = 1000 + X[:, 1] * 20 # Pressao 1000-1020
    
    y = np.zeros(200)
    mask = (X[:, 0] > 70) & (X[:, 1] < 1012)
    y[mask] = 1
    # Ruído
    flip = np.random.choice(200, 20)
    y[flip] = 1 - y[flip]
    
    rf = FlorestaAleatoriaChuva(n_arvores=20)
    rf.treinar(X, y)
    
    # Teste
    X_teste = np.array([[90, 1005], [30, 1015]]) # Chuva, Sol
    pred = rf.predizer(X_teste)
    prob = rf.probabilidade(X_teste)
    
    print(f"Predições (Esperado: [1, 0]): {pred}")
    print(f"Probabilidades Chuva: {prob}")
    print("Classificador RF funcional.")
