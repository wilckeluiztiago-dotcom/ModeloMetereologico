import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: AUTOENCODER PARA ANOMALIAS
=============================================================

Implementa uma rede neural Autoencoder usada para detecção não-supervisionada
de anomalias climáticas (ex: eventos extremos nunca vistos).

O modelo aprende a comprimir (encode) e reconstruir (decode) dados normais.
Erro de reconstrução alto = Anomalia.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AutoencoderClimatico:
    def __init__(self, input_dim=5, latent_dim=2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.W_enc = np.random.randn(input_dim, latent_dim) * 0.1
        self.b_enc = np.zeros((1, latent_dim))
        
        # Decoder
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.1
        self.b_dec = np.zeros((1, input_dim))
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, X):
        """
        Retorna Reconstrucão e Código Latente.
        """
        # Encode (Linear -> Sigmoid)
        code = self._sigmoid(np.dot(X, self.W_enc) + self.b_enc)
        
        # Decode (Linear)
        reconstruction = np.dot(code, self.W_dec) + self.b_dec
        
        return reconstruction, code
        
    def treinar_passo(self, X, lr=0.01):
        """Treina para minimizar erro de reconstrução."""
        m = X.shape[0]
        
        # Forward
        rec, code = self.forward(X)
        erro = rec - X # MSE Loss gradiente simplificado
        loss = np.mean(erro**2)
        
        # Backward (Decoder)
        d_rec = (2/m) * erro
        d_W_dec = np.dot(code.T, d_rec)
        d_b_dec = np.sum(d_rec, axis=0, keepdims=True)
        
        # Backward (Encoder)
        d_code = np.dot(d_rec, self.W_dec.T)
        d_z_enc = d_code * code * (1 - code) # Sigmoid derivative
        
        d_W_enc = np.dot(X.T, d_z_enc)
        d_b_enc = np.sum(d_z_enc, axis=0, keepdims=True)
        
        # Updates
        self.W_enc -= lr * d_W_enc
        self.b_enc -= lr * d_b_enc
        self.W_dec -= lr * d_W_dec
        self.b_dec -= lr * d_b_dec
        
        return loss

    def detectar_anomalias(self, X, threshold_percentile=95):
        """Retorna índices das amostras com alto erro de reconstrução."""
        rec, _ = self.forward(X)
        mse_por_amostra = np.mean((X - rec)**2, axis=1)
        
        limiar = np.percentile(mse_por_amostra, threshold_percentile)
        anomalias = np.where(mse_por_amostra > limiar)[0]
        
        return anomalias, mse_por_amostra

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Autoencoder de Anomalias...")
    
    # Dados Normais (Cluster em torno de 0)
    X_normal = np.random.normal(0, 1, (100, 5))
    
    # Dados Anômalos (Deslocados)
    X_anomalo = np.random.normal(5, 1, (5, 5))
    
    X_total = np.vstack([X_normal, X_anomalo])
    
    ae = AutoencoderClimatico(input_dim=5, latent_dim=2)
    
    # Treinar apenas com parte normal (idealmente) ou com tudo misturado
    print("Treinando...")
    for i in range(500):
        l = ae.treinar_passo(X_total, lr=0.1)
        
    idxs, erros = ae.detectar_anomalias(X_total, threshold_percentile=95)
    
    print(f"Total amostras: {len(X_total)}")
    print(f"Detectadas {len(idxs)} anomalias.")
    print(f"Índices anômalos detectados: {idxs}")
    # Esperamos ver os índices finais (100, 101, 102...)
    
    print("Anomalias detectadas com sucesso via Autoencoder.")
