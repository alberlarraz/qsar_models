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
- No hay script. Se recomienda hacer de forma manual en excel
- Los endpoints con un índice de correlación superior al umbral deben ser eliminados hasta que solo queden descriptores independientes entre sí.
- Este umbral mínimo estará entre 0.5 y 0.9
- Recomendación para la selección de descriptores:
        · Colocar aplicados en columnas cada uno de los descriptores con un valor de correlación (uso de macro de excel)
        · Eliminar manualmente los descriptores tomando decisiones basadas en los siguientes criterios hasta que solo quede el nombre del mismo compuesto bajo la columna:
              *Criterio 1: Aquellos descriptores con mayor número de correlaciones deberían ser eliminados primero
              *Criterio 2: Aquellos descriptores con significado químico claro deberían prevalecer frente a complejos algortimos
              *Criterio 3: Se recomienda hacer selecciones diferentes para probar en los pasos posteriores.

STEP 4/ CLUSTER ANALYSIS
- Una vez hechas las seleciones de los descriptores, se seleccionan las columnas de la hoja de excel no elegidas para eliminarlas.
- Se deben escribir los descriptores elegidos en la lista data_alergenos_numeric.
- Se debe elegir los parámetros de clusterización: Tipo de distancia y de método de enlace (Recomendación: Euclidean+Complete)
- Este paso tiene dos objetivos: Eliminación de outliers debido a estar muy separados del resto del dominio de aplicabilidad & Poder separar en test set y training set de manera homogénea

STEP 5/ TRAINING SET VS TEST SET
- El excel creado en el paso 4 "data_alergenos_numeric" selecciona un número de cluster para cada compuesto.
- La división test set/training set debe estar entorno al 20-30/80-70%. Por ello, esta división debe estar hecha dentro de cada uno de los clusters.
- Para ello, se asigna un número aleatorio mediante la función random de excel entre 0-1. Test set < 0.2-0.3 < Training Set (Debe asegurarse de que el cómputo global cumple este reparto)

STEP 6/ FITTING ALGORTITHMS
- Este script contiene un algoritmo de regresión lineal múltiple en combinación con un algoritmo genético para seleccionar la mejor combinación:
          · 1/ Se importan los datos desde un excel con el training set y otro con el test set
          · 2/ Se indican los descriptores elegidos en el paso 3
          · 3/ Normalización de los datos. Todos los descriptores deben estar en los mismos rangos (0-1) para tener el mismo peso.
          · 4/ COMIENZA EL ALGORITMO GENÉTICO:
                  *individual_size = ¿De cuantos endpoints va a consistir mi modelo?
                  *population_size = Número inicial de combinaciones aleatorias
                  *selection_percentage = Me quedo con las mejores para el siguiente paso (Cuantas?)
                  *selection_size = Numero de combinaciones que continuan
                  *max_generations = ¿Cuantas veces voy a hacer esto? -> Hay un momento en el que ya apenas mejora
                  *probability_of_individual_mutating = ¿Cuantas combinaciones van a mutarse?
                  *probability_of_gene_mutating = Descriptores mutados dentro de cada selección
                  *initial_population = Numero de combinaciones iniciales
                  *Clasificación según cross_val_score
                  *COMBINACIONES DE DESCRIPTORES -> HACEMOS CON TODAS EL MLR MODEL -> NOS QUEDAMOS CON LAS MEJORES -> LAS RECOMBINAMOS -> MUTAMOS UNAS CUANTAS -> VUELTA A EMPEZAR
          · 5 / Máximo de generaciones: Clasificamos y da los mejores 10 resultados

STEP 7/ REPRESENTATION 
- Selección de los descriptores con mejor resultado en el paso anterior
- Input data: Training set
- Hace de nuevo el MLR y te lo representa, dandote los estadísticos
- Una vez hecho esto habría que representar el test_set, para comprobar la calidad de la validación externa
