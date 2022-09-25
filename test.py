from rdkit import Chem
from conf_calc import ConfCalc

mol_file = "test.mol"
mol = Chem.MolFromMolFile(mol_file, removeHs=False)

calculator = ConfCalc(mol=mol,
                      rotable_dihedral_idxs=[[5, 4, 6, 7], 
                                             [4, 6, 7, 9],
                                             [12, 13, 15, 16]])
print(calculator.get_energy([0., 0., 3.14]))
print(calculator.get_energy([0., 0., 0.]))
