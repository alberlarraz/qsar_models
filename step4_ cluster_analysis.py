#STEP 1: IMPORT LIBRARIES 

import pandas as pd
import numpy as np
from pylab import figure, colorbar
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import whiten
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

#STEP 2: UPLOAD DATA FROM AN EXCEL FILE

pd.options.display.float_format = "{:,.1f}".format
data_alergenos = pd.read_excel("descriptores_posproc.xlsx")
data_alergenos.index = data_alergenos["COMP"]
data_alergenos_numeric = data_alergenos[["BCUT.4", "BCUT.2", "petitjeanShapeIndex.1", "CPSA.21", "WHIM.3", "MaxEStateIndex", "Weta2.unity", "WNSA-1", "Wlambda2.unity",
 "types.GeometricalRadius", "WD.unity", "ALogP", "ALogp2", "ATSC3c", "ATSC1e", "ATSC3i", "AATSC3m", "AATSC2i", "MATS2m", "MATS1s", "MATS3s", "GATS1c", "GATS2c", "GATS3e",
  "VE3_Dzp", "VR3_Dzi", "SM1_Dzs", "VR1_Dzs", "SpMin5_Bhs", "AVP-1", "gmin", "BIC3", "TDB3m", "TDB3i", "WPSA-326", "RNCS33", "GRAV-444", "RDF35u", "RDF15s", "E1m", "Dv", "De",
   "E2p", "E2i", "E1s", "Ks"
]].astype(float)

#STEP 3: USE OF WHITEN FUNCTION TO HAVE THE SAME STANDARD DESVIATION FOR EACH COLUMN -> Equal influence

whiten_numeric = pd.DataFrame(data = whiten(obs = data_alergenos_numeric, check_finite = True), columns=data_alergenos_numeric.columns)
whiten_numeric.index = data_alergenos_numeric.index
#print(whiten_numeric)
#for j in whiten_numeric.columns:
#   print([j, whiten_numeric[j].std()])

#STEP 4.1: DISTANCE METRIC (Distance matrix with Euclidian distance)

dict_pdist = {}
list_metric = [
    "euclidean", 
    "cityblock", 
    "correlation", 
    "cosine"
]
for j in list_metric:
    dict_pdist[j] = pdist(
        X=whiten_numeric, 
        metric = j
)
#print(dict_pdist)
#print(squareform(X=dict_pdist["euclidean"])) # -> To see how the whole matrix looks like

#STEP 4.2: DISTANCE METRIC (Method Linkage: Complete)
dict_linkage = {}
list_method = [
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward"
]
for j in list_metric:
    for k in list_method:
        dict_linkage[j + "_" + k] = linkage(
            y = dict_pdist[j],
            method = k,
            metric = j,
            optimal_ordering = True
        )
#print(dict_linkage["euclidean_complete"])

#STEP 5: PLOT THE DENDROGRAM

fig = plt.figure(figsize=(18,15))
dendrogram(
    Z = dict_linkage['euclidean_complete'],
    labels = whiten_numeric.index,
    count_sort = True,
    distance_sort= True,
    orientation= "right",
    leaf_font_size= 8
)

plt.show()

#STEP 6: SAVE THE DATA
threshold= input("Threshold: ")
clusters = fcluster(Z=dict_linkage['euclidean_complete'], t=float(threshold), criterion='distance')
data_alergenos_numeric['Cluster'] = clusters
data_alergenos_numeric['COMP'] = data_alergenos["COMP"]
data_alergenos_numeric.to_excel('alergenos_clusters_euclidean_complete.xlsx', index=False)