# qsar_models
Step by step process to create your own QSAR model from a database

STEP 1/ MOLECULAR DESCRIPTORS
- Cálculo de descriptores moleculares desde 3 librerías diferentes: RDKIT, MORDRED y PADELPY.
- Primero se compruban los SMILES y se transforman a SMILES canónicos para evitar problemas al calcular los descriptores
- Transformamos los SMILES en archivos MOL para crear moléculas
- Calculamos los descriptores. ¡OJO! Mordred suele dar problemas. Padelpy tiene descriptores más interesantes, pero en ocasiones luego es dificil acceder a ellos. RDKIT siempre funciona

STEP 2/ CORRELATION MATRIX


STEP 3/ SELECTION OF INDEPENDENT DESCRIPTORS


STEP 4/ CLUSTER ANALYSIS


STEP 5/ TRAINING SET VS TEST SET


STEP 6/ FITTING ALGORTITHMS


STEP 7/ REPRESENTATION 
