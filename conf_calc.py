import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import copy

class ConfCalc:

    def __init__(self, 
                 mol_file_name : str = None,
                 mol : Chem.Mol = None,
                 rotable_dihedral_idxs : list[list[int]] = None):
        """
            Class that calculates energy of current molecule
            with selected values of dihedral angles
            mol_file_name - path to .mol file
            mol - Chem.Mol object
            rotable_dihedral_idxs - list of 4-element-lists with 
            integer zero-numerated indexes 
        """        
        
        assert (mol_file_name is not None or mol is not None),\
             """No mol selected!"""

        assert rotable_dihedral_idxs is not None,\
             """No idxs to rotate have been selected!"""

        if mol_file_name is None:
            self.mol = mol
        elif mol is None:
            self.mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)  

        self.rotable_dihedral_idxs = rotable_dihedral_idxs

    def __setup_dihedrals(self,
                          values : list[float]) -> Chem.Mol:
        """
            Private function that returns a molecule with
            selected dihedral angles
            values - list of angles in radians
        """
        
        assert len(values) == len(self.rotable_dihedral_idxs),\
             """Number of values must be equal to the number of
                dihedral angles"""
        
        new_mol = copy.deepcopy(self.mol)        

        for idxs, value in zip(self.rotable_dihedral_idxs, values):
            rdMolTransforms.SetDihedralRad(new_mol.GetConformer(), *idxs, value)
        
        return new_mol

    def __calc_energy(self, 
                      mol : Chem.Mol) -> float:
        """
            Calculates energy of given molecule via xtb
        """
        
        return np.random.rand()

    def get_energy(self, 
                   values : list[float]) -> float:
        """
            Returns energy of molecule with selected values
            of dihedral angles    
        """

        mol = self.__setup_dihedrals(values)
        return self.__calc_energy(mol)
