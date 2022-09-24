from rdkit import Chem
from conf_calc import ConfCalc

mol_file = "test.mol"
mol = Chem.MolFromMolFile(mol_file)

calculator = ConfCalc(mol=mol,
                      rotable_dihedral_idxs=[[0, 1, 2, 3], 
                                             [1, 2, 3, 4]])
print(calculator.get_energy([1., 2.3]))
