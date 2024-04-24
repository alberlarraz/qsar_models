#IMPORTAR PANDAS COMO LIBRERÍA
import pandas as pd

#IMPORTAR EXCEL CON LOS DESCRIPTORES
#Respecto al excel obtenido en el anterior paso deben hacerse los siguientes cambios
#La primera columna contendrá el nombre de las moléculas. Recomendado nombrarlas como C1, C2, C3, ..., Cn
#El resto de columnas deben ser los resultados de todos los descriptores. La primera fila queda reservada para el nombre de los descriptores.
dataset = pd.read_excel('descriptores_posproc.xlsx')
df = pd.DataFrame(dataset)

#CÁLCULO DE LA MATRIZ DE CORRELACIÓN
correlation_matrix = df.corr()

#IMPRIMIR LA MATRIZ DE CORRELACIÓN
correlation_matrix.to_csv("correlation_matrix_act.csv", index=False)
print("MATRIZ DE CORRELACIÓN FINALIZADA")
