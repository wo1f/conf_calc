import numpy as np
from rdkit import Chem
from conf_calc import ConfCalc

mol_file = "test.mol"
mol = Chem.MolFromMolFile(mol_file, removeHs=False)

calculator = ConfCalc(mol=mol,
                      dir_to_xyzs="xtb_calcs/",
                      rotable_dihedral_idxs=[[5, 4, 6, 7], 
                                             [4, 6, 7, 9],
                                             [12, 13, 15, 16]])
print(calculator.log_prob(np.array([0., 0., 0.])))
print(calculator.log_prob_grad(np.array([0., 0., 0.])))
# print(calculator.get_energy(np.array([0., 0., 0.])))
# print(calculator.get_energy(np.array([0., 0., 3.14])))
# print(calculator.get_energy(np.array([3.14, 0, 3.14]), req_opt=False, req_grad=False))

