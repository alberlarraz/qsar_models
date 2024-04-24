from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import pandas as pd

dataset = pd.read_csv('alergenos.csv', delimiter=";")

#FÓRMULA PARA CAMBIAR A SMILES CANONICOS
def canonical_smiles(smiles):
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in molecules]
    return smiles

#TRANSFORMAMOS NUESTROS SMILES EN SMILES CANONICOS
Canon_SMILES = canonical_smiles(dataset.SMILES)
num = len(Canon_SMILES)

writer1 = Chem.SDWriter('alergenos_confs_1_60.sdf')
writer2 = Chem.SDWriter('alergenos_confs_61_120.sdf')
writer3 = Chem.SDWriter('alergenos_confs_121_180.sdf')

for count, molecule in enumerate(Canon_SMILES):
    mol = Chem.AddHs(Chem.MolFromSmiles(molecule))
    conformers = AllChem.EmbedMultipleConfs(mol, numConfs=8)
    conformer_i = mol.GetConformer(0)  # This is the conformer that I wanted to save in a SDF file.
    #print(conformer_i.GetPositions())
    mol.SetProp("Name", dataset.COMP[count])  # Agrega el nombre del compuesto
    mol.SetProp("SMILES", molecule)  # Agrega el SMILES original

    if count < 60:
        writer1.write(mol, confId=0)
    elif count < 120:
        writer2.write(mol, confId=0)
    elif count < 180:
        writer3.write(mol, confId=0)

    print(count, "of", num - 1)

writer1.close()
writer2.close()
writer3.close()