import numpy as np
from typing import Optional

from rdkit import Chem
from conf_calc import ConfCalc


def get_prepared_calculator(mol_file, zero_values: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0]), clear_cache=True):
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    calculator = ConfCalc(mol=mol,
                          dir_to_xyzs="xtb_calcs_one/",
                          clear_cache=clear_cache,
                          rotable_dihedral_idxs=[[5, 4, 6, 7],
                                                 [4, 6, 7, 9],
                                                 [12, 13, 15, 16]])

    if zero_values is not None:
        zero_level = calculator.log_prob(zero_values)
        calculator.norm_en = zero_level

    return calculator


if __name__ == '__main__':
    import sampler_one
    sampler_one.main(get_prepared_calculator("test.mol"))
