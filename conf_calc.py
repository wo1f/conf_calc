import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import copy
import os
import time

class ConfCalc:

    def __init__(self, 
                 mol_file_name : str = None,
                 mol : Chem.Mol = None,
                 rotable_dihedral_idxs : list[list[int]] = None,
                 dir_to_xyzs : str = "", 
                 charge : int = 0,
                 gfn_method : int = 2,
                 timeout : int = 250,
                 norm_en : int = 0.):
        """
            Class that calculates energy of current molecule
            with selected values of dihedral angles
            mol_file_name - path to .mol file
            mol - Chem.Mol object
            rotable_dihedral_idxs - list of 4-element-lists with 
            integer zero-numerated indexes
            dir_to_xyzs - path of dir, where .xyz files will be saved
            charge - charge of molecule
            gfn_method - type of GFN method
            timeout - period of checking log file
            norm_en - norm energy 
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
        
        if dir_to_xyzs != "":
            dir_to_xyzs = dir_to_xyzs if dir_to_xyzs[-1] == "/" else dir_to_xyzs + "/"

        self.dir_to_xyzs = dir_to_xyzs

        self.charge = charge
        self.gfn_method = gfn_method
        self.timeout = timeout
        self.norm_en = norm_en

        # Id of next structure to save
        self.current_id = 0

    def set_norm_en(self,
                    norm_en = 0.):
        """
            Updates norm energy
        """

        self.norm_en = norm_en

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

    def __save_mol_to_xyz(self, 
                          mol : Chem.Mol) -> str:
        """
            Saves given mol to file, returns name of file
        """
        
        file_name = self.dir_to_xyzs + str(self.current_id) + ".xyz"
        self.current_id += 1
        Chem.MolToXYZFile(mol, file_name)

        return file_name

    def __run_xtb(self,
                  xyz_name : str) -> str:
        """
            Runs xtb with current xyz_file, returns name of log file
            timeout - period im ms to check log file
        """

        log_name = xyz_name[:-3] + "log"
        os.system(f"xtb --charge {self.charge} --gfn {self.gfn_method} {xyz_name} > {log_name}")
        
        while True:
            try:
                with open(log_name, "r") as file:
                    line_with_en = [
                        line for line in file
                            if "TOTAL ENERGY" in line
                    ]
                    if len(line_with_en) != 0:
                        return log_name
            except FileNotFoundError:
                pass
            finally:
                time.sleep(self.timeout / 1000)

    def __parse_energy_from_log(self, 
                                log_name : str) -> float:
        """
            Gets energy from xtb log file
        """

        energy = 0.
        with open(log_name, "r") as log_file:
            energy = [
                line for line in log_file 
                    if "TOTAL ENERGY" in line
                ][0].split()[3]

        return float(energy)

    def __calc_energy(self, 
                      mol : Chem.Mol) -> float:
        """
            Calculates energy of given molecule via xtb
        """
        xyz_name = self.__save_mol_to_xyz(mol)
        log_name = self.__run_xtb(xyz_name)

        return self.__parse_energy_from_log(log_name)

    def get_energy(self, 
                   values : list[float]) -> float:
        """
            Returns energy of molecule with selected values
            of dihedral angles    
        """

        mol = self.__setup_dihedrals(values)
        return self.__calc_energy(mol) - self.norm_en
