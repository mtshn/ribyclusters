import pandas as pd
import rdkit
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


desc_names= open("rdkit_descriptors.txt",'r').read().splitlines()
smis= open("tmpForRDKit_" + sys.argv[1] + ".txt",'r').read().splitlines()

calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
headr = calc.GetDescriptorNames()
descriptors = []

i = 0
for smi in smis:
      ds = calc.CalcDescriptors(Chem.MolFromSmiles(smi))
      descriptors.append(ds)
      if i % 1000 == 0:
          print('Calculating RDKit descriptors: '+str(i))
      i = i+1

df = pd.DataFrame(descriptors,columns=headr)
df.to_csv("tmpForRDKit_out_" + sys.argv[1] + ".txt", index=False)

print(rdkit.__version__)
