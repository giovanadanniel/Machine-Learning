import matplotlib.pyplot as plt
import numpy as np

# Dados do gráfico
restaurantes = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
real_evaluations = [10, 15, 8, 20, 12, 18, 22, 25, 17, 19]
surrogate_evaluations = [4, 5, 4, 7, 6, 5, 6, 7, 5, 6]

# Configurando o gráfico
x = np.arange(len(restaurantes))
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, real_evaluations, width=0.4, label="Avaliação Direta", color="blue")
plt.bar(x + 0.2, surrogate_evaluations, width=0.4, label="Com Surrogate", color="red")
plt.title("Avaliações Necessárias: Diretas vs Surrogate")
plt.xlabel("Restaurantes")
plt.ylabel("Número de Avaliações")
plt.xticks(x, restaurantes)
plt.legend()
plt.grid(axis='y', linestyle="--")

# Salvando o gráfico
plt.savefig("avaliacoes_surrogate_final.png")
plt.show()
