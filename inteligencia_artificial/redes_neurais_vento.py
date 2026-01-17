import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: REDES NEURAIS (MLP) PARA VENTO
=================================================================

Implementa um Perceptron Multicamadas (Feedforward Neural Network)
para estimativa de velocidade do vento baseada em gradientes de pressão e temperatura.

Arquitetura:
- Input Layer: [Gradiente Pressão X, Gradiente Pressão Y, Diferença Temp, Rugosidade]
- Hidden Layer: Neurônios com ativação ReLU
- Output Layer: Velocidade do Vento Escalar

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class MLPRegressorVento:
    def __init__(self, input_dim=4, hidden_dim=8):
        # Inicialização He
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros((1, 1))
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        """
        X: matriz (n_samples, input_dim).
        Retorna predições (n_samples, 1).
        """
        # Camada Oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        
        # Camada Saída (Linear para regressão)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
        
    def treinar_lote(self, X, y, lr=0.001):
        """
        Executa um passo de treinamento (Backpropagation) num mini-batch.
        """
        m = X.shape[0]
        
        # Forward
        preds = self.forward(X)
        erro = preds - y.reshape(-1, 1)
        loss = np.mean(erro**2)
        
        # Backward
        d_z2 = (2/m) * erro
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0) # Derivada ReLU
        
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Update
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        
        return loss

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Rede Neural de Vento (MLP)...")
    
    # Gerar dados sintéticos (Física aproximada: Vento ~ Gradiente Pressão)
    n = 1000
    grad_p_x = np.random.randn(n) * 2
    grad_p_y = np.random.randn(n) * 2
    
    # Target: Vento Geostrófico + Ruído
    vento_real = np.sqrt(grad_p_x**2 + grad_p_y**2) * 5 + np.random.normal(0, 0.5, n)
    
    # Input
    X = np.column_stack((grad_p_x, grad_p_y, np.abs(grad_p_x), np.abs(grad_p_y)))
    y = vento_real
    
    modelo = MLPRegressorVento(input_dim=4, hidden_dim=16)
    
    loss_inicial = modelo.treinar_lote(X, y) # Teste inicial
    
    print(f"Loss Inicial: {loss_inicial:.4f}")
    
    # Loop treino rápido
    for i in range(100):
        l = modelo.treinar_lote(X, y, lr=0.01)
        if i % 20 == 0: print(f"Epoca {i}: Loss {l:.4f}")
        
    print("Treinamento concluído.")
    
    # Validar
    teste = np.array([[1.0, 0.0, 1.0, 0.0]]) # Gradiente moderado em X
    pred = modelo.forward(teste)[0][0]
    print(f"Estimativa para grad=1 (Esperado ~5 m/s): {pred:.2f} m/s")
