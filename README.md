## ConfCalc

This script provides you with the ability to calculate the energy of a molecule with variable dihedral angles via xtb.

##### How to use:

Firstly, you should create object of ConfCalc class and define some properties:

```python
    from conf_calc import ConfCalc

    calculator = ConfCalc(mol_file_name="test.py", 
                          rotable_dihedral_idxs=[[0, 1, 2, 3],
                                                 [1, 2, 3, 4]])
```

or

```python
    from conf_calc import ConfCalc

    calculator = ConfCalc(mol=some_mol, 
                          rotable_dihedral_idxs=[[0, 1, 2, 3],
                                                  1, 2, 3, 4])
```

Where *some_mol* - rdkit.Chem.Mol object, *rotable_dihedral_idxs* - list of idxs of atoms in rotable dihedrals.
Next, to get energy of current conformer just type:

```python
    calculator.get_energy([0., np.pi / 2])
```

##### Requirements:

* rdkit
