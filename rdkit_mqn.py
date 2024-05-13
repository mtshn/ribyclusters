import pandas as pd
import rdkit
import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


smis= open("tmpForRDKit_" + sys.argv[1] + ".txt",'r').read().splitlines()

descriptors = []

i = 0
for smi in smis:
      try:
         ds = rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smi))
         descriptors.append(ds)
      except:
         ds = [-777.0 for i in range(42)]
         descriptors.append(ds)

      if i % 1000 == 0:
          print('Calculating RDKit MQN descriptors: '+str(i))
      i = i+1

df = pd.DataFrame(descriptors)
df.to_csv("tmpForRDKit_out_" + sys.argv[1] + ".txt", index=False)

print(rdkit.__version__)
