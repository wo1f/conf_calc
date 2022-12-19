import pickle
import shutil

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from wilson_b_matrix import Dihedral, get_current_derivative

import subprocess
import copy
from typing import Tuple
from pathlib import Path


class ConfCalc:

    def __init__(self,
                 mol_file_name: str = None,
                 mol: Chem.Mol = None,
                 rotable_dihedral_idxs: list[list[int]] = None,
                 dir_to_xyzs: str = "",
                 charge: int = 0,
                 gfn_method: int = 2,
                 norm_en: float = 0.,
                 clear_cache=True):
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
            norm_en - norm energy
        """

        assert (mol_file_name is not None or mol is not None), \
            """No mol selected!"""

        assert rotable_dihedral_idxs is not None, \
            """No idxs to rotate have been selected!"""

        if mol_file_name is None:
            self.mol = mol
        elif mol is None:
            self.mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)

        self.rotable_dihedral_idxs = rotable_dihedral_idxs

        self.dir_to_xyzs = Path(dir_to_xyzs)
        if clear_cache:
            shutil.rmtree(self.dir_to_xyzs, ignore_errors=True)
            self.dir_to_xyzs.mkdir(exist_ok=True)

        self.charge = charge
        self.gfn_method = gfn_method
        self.norm_en = norm_en

        # Id of next structure to save
        self.dims = 3
        self.coef = 1045.8491666666666  # хартри в ккал\моль и поделить на RT
        self.xtb_args = ['xtb',
                         '--charge', str(self.charge),
                         '--gfn', str(self.gfn_method),
                         '--opt',
                         '--grad',
                         '--namespace',
                         ]

    def __setup_dihedrals(self, values: np.ndarray) -> Chem.Mol:
        """
            Private function that returns a molecule with
            selected dihedral angles
            values - list of angles in radians
        """

        assert len(values) == len(self.rotable_dihedral_idxs), \
            """Number of values must be equal to the number of
               dihedral angles"""

        new_mol = copy.deepcopy(self.mol)

        for idxs, value in zip(self.rotable_dihedral_idxs, values):
            rdMolTransforms.SetDihedralRad(new_mol.GetConformer(), *idxs, value)

        return new_mol

    def __save_mol_to_xyz(self, mol: Chem.Mol, xyz_name) -> Path:
        """
            Saves given mol to file, returns name of file
        """

        Chem.MolToXYZFile(mol, str(xyz_name))

        return xyz_name

    def __generate_name(self, values: np.ndarray) -> Path:
        return self.dir_to_xyzs / f"{values[0]}_{values[1]}_{values[2]}.inp"

    def __generate_inp(self, values: np.ndarray) -> Path:
        """
            Generate input that constrains values
            of dihedral angles from 'vals'
            Returns name of inp file
        """

        inp_name = self.__generate_name(values)

        with open(inp_name, "w+") as inp_file:
            inp_file.write("$constrain\n")
            for i, idxs in enumerate(self.rotable_dihedral_idxs):
                degree = (180 * values[i] / np.pi)
                inp_file.write(
                    f"dihedral: {idxs[0] + 1}, {idxs[1] + 1}, {idxs[2] + 1}, {idxs[3] + 1}, {degree}\n")
            inp_file.write("$end")
        return inp_name

    def __run_xtb(self,
                  xyz_name: Path,
                  inp_name: Path) -> float:
        """
            Runs xtb with current xyz_file, returns name of log file
            xyz_name - name of .xyz file
            inp_name - name of .inp file
        """

        output = subprocess.run(self.xtb_args + [xyz_name.stem, xyz_name.name, '--input', inp_name.name],
                                capture_output=True, text=True, cwd=self.dir_to_xyzs)

        return ConfCalc.__parse_energy_from_str(output.stdout)

    @staticmethod
    def __parse_energy_from_str(content: str) -> float:
        """
            Gets energy from xtb log file
        """
        for line in reversed(content.split('\n')):
            if line.startswith('          | TOTAL ENERGY'):
                return float(line.split()[3])

        return float(0)

    def __parse_grads_from_grads_file(self,
                                      num_of_atoms: int,
                                      namespace: str) -> np.ndarray:
        """
            Read gradinets from file, returns ['num_of_atoms', 3] numpy array with
            cartesian energy derivatives
        """
        grads = []
        with open(self.dir_to_xyzs / f'{namespace}.gradient', 'r') as file:
            grads = [line[:-1] for line in file][(2 + num_of_atoms):-1]
        return np.array(list(map(lambda s: list(map(float, s.split())), grads)))

    def __cart_grads_to_irc_grads(self, cart_grads, mol):
        irc_grad = np.empty(len(self.rotable_dihedral_idxs))
        for i, rotable_idx in enumerate(self.rotable_dihedral_idxs):
            irc_grad[i] = get_current_derivative(mol,
                                                 cart_grads,
                                                 Dihedral(*rotable_idx))
        return irc_grad

    def __calc_energy(self,
                      mol: Chem.Mol,
                      inp_name: Path) -> tuple:
        """
            Calculates energy of given molecule via xtb
            inp_name - name of file with input
            retruns tuple of energy and gradient
        """

        xyz_name = inp_name.with_suffix('.xyz')
        self.__save_mol_to_xyz(mol, xyz_name)
        energy = self.__run_xtb(xyz_name, inp_name)

        cart_grads = self.__parse_grads_from_grads_file(len(mol.GetAtoms()), inp_name.stem)
        irc_grad = self.__cart_grads_to_irc_grads(cart_grads.flatten(), mol)

        return energy, irc_grad

    def log_prob_one(self, values: np.ndarray) -> float:
        energy, _ = self.get_energy_one(values)
        energy *= self.coef
        return energy

    def log_prob_grad_one(self, values: np.ndarray) -> np.ndarray:
        _, grads = self.get_energy_one(values)
        grads *= self.coef
        return grads[2:]

    def log_prob(self, values: np.ndarray) -> float:
        energy, _ = self.get_energy(values)
        energy *= self.coef
        return energy

    def log_prob_grad(self, values: np.ndarray) -> np.ndarray:
        _, grads = self.get_energy(values)
        grads *= self.coef
        return grads

    @staticmethod
    def get_energy_from_files(inp_name: Path) -> Tuple[float, np.ndarray]:
        gradient = np.load(inp_name.with_suffix('.npy'))
        energy = pickle.loads(inp_name.with_suffix('.log').read_bytes())
        return energy, gradient

    def get_energy_one(self, values: np.ndarray) -> Tuple[float, np.ndarray]:
        vals = np.array([0.0, 0.0, 0.0])
        vals[2] = values[0]
        return self.get_energy(vals)

    def get_energy(self, values: np.ndarray) -> Tuple[float, np.ndarray]:
        """
            Returns dict with fields:
            'energy' - energy in this point
            'grads' - list of tuples, consists of
            pairs of dihedral angle atom indexes and
            gradients of energy with resoect to this angle
        """
        inp_name = self.__generate_name(values)
        if inp_name.with_suffix('.log').exists():
            return self.get_energy_from_files(inp_name)

        inp_name = self.__generate_inp(values)
        mol = self.__setup_dihedrals(values)

        energy, gradient = self.__calc_energy(mol, inp_name)
        energy -= self.norm_en

        np.save(inp_name.with_suffix(''), gradient)
        inp_name.with_suffix('.log').write_bytes(pickle.dumps(energy))
        # print(values, energy)
        return energy, gradient
