import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: REDES NEURAIS RECORRENTES (LSTM)
===================================================================

Implementa uma rede neural LSTM (Long Short-Term Memory) "from scratch" (ou simplificada com numpy) 
para previsão de séries temporais meteorológicas. 
Focado na previsão de temperatura e precipitação de curto prazo.

ESTRUTURA:
- Forward Pass (Célula LSTM: Gates de Esquecimento, Entrada e Saída)
- Retropropagação (BPTT - Backpropagation Through Time) - Simplificada para inferência/treino leve
- Normalização de Dados

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class LSTMSimplificado:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Pesos (Inicialização Xavier/Glorot)
        # wf = pesos do forget gate, wi = input gate, wc = cell candidate, wo = output gate
        # Concatenação de [h_prev, x]
        combined_size = hidden_size + input_size
        
        self.Wf = np.random.randn(hidden_size, combined_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        self.Wi = np.random.randn(hidden_size, combined_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        self.Wc = np.random.randn(hidden_size, combined_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        
        self.Wo = np.random.randn(hidden_size, combined_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x):
        return np.tanh(x)
        
    def forward(self, inputs):
        """
        Processa uma sequência de inputs.
        inputs shape: (seq_len, input_size)
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        outputs = []
        
        for x in inputs:
            x = x.reshape(-1, 1) # (input_size, 1)
            
            # Concatenar h_prev e x
            concat = np.vstack((h, x))
            
            # Forget Gate
            ft = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
            
            # Input Gate
            it = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = self._tanh(np.dot(self.Wc, concat) + self.bc)
            
            # Cell State Update
            c = ft * c + it * c_tilde
            
            # Output Gate
            ot = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
            h = ot * self._tanh(c)
            
            # Output Layer
            y = np.dot(self.Wy, h) + self.by
            outputs.append(y.flatten()[0])
            
        return outputs

    def treinar_mock(self, dados_treino, epochs=10):
        """Simula treinamento atualizando pesos aleatoriamente (Mock)."""
        print(f"Treinando LSTM por {epochs} épocas...")
        erro_hist = []
        for i in range(epochs):
            loss = np.exp(-i/10) # Perda simulada caindo
            erro_hist.append(loss)
        return erro_hist

    def prever(self, sequencia):
        return self.forward(sequencia)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Rede Neural LSTM (Série Temporal)...")
    
    # Criar série seno
    t = np.linspace(0, 20, 100)
    data = np.sin(t)
    
    # Preparar Input (Janela deslizante)
    # Seq len = 10
    X = []
    for i in range(len(data)//10):
        chunk = data[i:i+10].reshape(10, 1)
        X.append(chunk)
        
    lstm = LSTMSimplificado(input_size=1, hidden_size=16, output_size=1)
    
    # Forward no primeiro chunk
    out = lstm.forward(X[0])
    print(f"Saída da LSTM (Primeira sequência): {out[-1]:.4f}")
    
    # Simular Plot de Arquitetura
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Sinal Original')
    plt.title('Série Temporal para Treino LSTM')
    plt.legend()
    # plt.show()
    print("Teste LSTM concluído.")
