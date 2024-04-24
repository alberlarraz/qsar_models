# qsar_models
Step by step process to create your own QSAR model from a database

STEP 1/ MOLECULAR DESCRIPTORS
- Cálculo de descriptores moleculares desde 3 librerías diferentes: RDKIT, MORDRED y PADELPY.
- Primero se compruban los SMILES y se transforman a SMILES canónicos para evitar problemas al calcular los descriptores
- Transformamos los SMILES en archivos MOL para crear moléculas
- Calculamos los descriptores. ¡OJO! Mordred suele dar problemas. Padelpy tiene descriptores más interesantes, pero en ocasiones luego es dificil acceder a ellos. RDKIT siempre funciona

STEP 2/ CORRELATION MATRIX
- Cálculo del indice de correlación entre los diferentes descriptores
- Adaptar el excel de input según las indicaciones

STEP 3/ SELECTION OF INDEPENDENT DESCRIPTORS
- Los endpoints con un índice de correlación superior al umbral deben ser eliminados hasta que solo queden descriptores independientes entre sí.
- Este umbral mínimo estará entre 0.5 y 0.9
- Recomendación para la selección de descriptores:
        · Colocar aplicados en columnas cada uno de los descriptores con un valor de correlación (uso de macro de excel)
        · Eliminar manualmente los descriptores tomando decisiones basadas en los siguientes criterios hasta que solo quede el nombre del mismo compuesto bajo la columna:
              *Criterio 1: Aquellos descriptores con mayor número de correlaciones deberían ser eliminados primero
              *Criterio 2: Aquellos descriptores con significado químico claro deberían prevalecer frente a complejos algortimos
              *Criterio 3: Se recomienda hacer selecciones diferentes para probar en los pasos posteriores.

STEP 4/ CLUSTER ANALYSIS


STEP 5/ TRAINING SET VS TEST SET


STEP 6/ FITTING ALGORTITHMS


STEP 7/ REPRESENTATION 
