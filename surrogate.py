import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados
x = np.array([0.01, 0.05, 0.1, 0.15]).reshape(-1, 1)  # Taxa de aprendizado
y = np.array([60, 75, 80, 70])  # Precisão

# Modelo de regressão linear
model = LinearRegression()
model.fit(x, y)

# Predições
x_pred = np.linspace(0, 0.2, 100).reshape(-1, 1)
y_pred = model.predict(x_pred)  # Modelo treinado para fazer previsões

# Gráfico
plt.scatter(x, y, color='blue', label='Dados Reais')  # Pontos reais
plt.plot(x_pred, y_pred, color='red', label='Modelo de Regressão')  # Linha de regressão
plt.xlabel('Taxa de Aprendizado')
plt.ylabel('Precisão (%)')
plt.title('Regressão Linear da Precisão em Função da Taxa de Aprendizado')
plt.legend()
plt.grid()
plt.show()
