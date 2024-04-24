import pandas as pd

# Importamos dataframe
dataset = pd.read_excel('descriptores_posproc.xlsx')
df = pd.DataFrame(dataset)

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Imprimir la matriz de correlación
correlation_matrix.to_csv("correlation_matrix_act.csv", index=False)
print("MATRIZ DE CORRELACIÓN FINALIZADA")