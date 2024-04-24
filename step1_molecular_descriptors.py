from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
from mordred import Calculator, descriptors
from padelpy import from_smiles
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
import csv

#LECTURA DEL CSV Y COMPROBAR EL NÚMERO DE FILAS Y COLUMNAS
dataset = pd.read_csv('Antifungica.csv', delimiter=";")
print("LECTURA COMPLETA")
print("Original dataset: ", dataset.shape)
#print(dataset.head())

#FÓRMULA PARA CAMBIAR A SMILES CANONICOS
def canonical_smiles(smiles):
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in molecules]
    return smiles

#TRANSFORMAMOS NUESTROS SMILES EN SMILES CANONICOS
Canon_SMILES = canonical_smiles(dataset.SMILES)
dataset['Smiles'] = Canon_SMILES
print("CANONICAL SMILES CALCULADOS")
print("Canonical SMILES dataset: ", dataset.shape)

'''
#ELIMINAMOS VALORES DUPLICADOS POR SI ACASO
duplicates_smiles = dataset[dataset['SMILES'].duplicated()]['SMILES'].values
print("DUPLICADOS EN LA DATABASE ELIMINADOS")
print("Number of duplicates: ", len(duplicates_smiles))
dataset_new = dataset.drop_duplicates(subset=['SMILES'])
print("Number of compounds on the new database: ", len(dataset_new))
'''

mols = [Chem.MolFromSmiles(smi) for smi in Canon_SMILES]

print("CALCULANDO DESCRIPTORES....")

#CÁLCULO DESCRIPTORES PADEL
calc_padel = from_smiles(Canon_SMILES, output_csv='padel_descriptors.csv')

print("PadelPy: OK")

#CÁLCULO DESCRIPTORES MORDRED
calc_mordred = Calculator(descriptors, ignore_3D=True)
df_mordred_desc = calc_mordred.pd(mols)
df_mordred_desc.to_csv("SMILES_new.csv", index=False)

print("Mordred: OK")

#CÁLCULO DESCRIPTORES RDKIT
calc_rdkit = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
desc_names = calc_rdkit.GetDescriptorNames()
mol_descriptors = []
for mol in mols:
    #Añadimos hidrógenos
    molH = Chem.AddHs(mol)
    #Calculamos todos los descriptores (200) para cada molécula
    rd_descriptors = calc_rdkit.CalcDescriptors(molH)
    mol_descriptors.append(rd_descriptors)

print("RDKit: OK")

df_with_rdkit_descriptors = pd.DataFrame(mol_descriptors,columns=desc_names)
df_with_rdkit_descriptors.to_csv("rdkit.csv", index=False)

print("ARCHIVOS CREADOs Y FINALIZADOs")