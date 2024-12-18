import numpy as np
import matplotlib.pyplot as plt


iterations = np.arange(1, 11) 
initial_exploration = [10, 12, 15, 13, 14, 16, 18, 17, 19, 20]  
surrogate_adjustment = [11, 13, 16, 15, 15.5, 17, 19, 18.5, 19.5, 20]  
final_result = [20] * 10  


plt.figure(figsize=(10, 6))
plt.plot(iterations, initial_exploration, label="Exploração Inicial", color="orange", linestyle="--", linewidth=2)
plt.plot(iterations, surrogate_adjustment, label="Ajuste do Surrogate", color="blue", linestyle="-", linewidth=2)
plt.plot(iterations, final_result, label="Resultado Final (Convergência)", color="green", linestyle=":", linewidth=2)
plt.title("Processo do Surrogate em Ação", fontsize=14)
plt.xlabel("Iterações", fontsize=12)
plt.ylabel("Qualidade da Solução", fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()


plt.savefig("processo_surrogate_otimizacao.png")
plt.show()
