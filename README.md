## ConfCalc

This script provides you with the ability to calculate the energy of a molecule with variable dihedral angles via xtb.

##### How to use:

For correct calculation, xtb should be in PATH.

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

It returns dictionary, where in field 'energy' will be current conformers' energy

To update normalization energy:

```python
calculator.set_norm_en(0.)
```

If you need energy derivatives with respect to torsion angles, type this:

```python
calculator.get_energy([0., 0., 0.], req_grad=True)
```

It will return dict with energy and list of gradients like this:

```python
{
    'energy': 0.,
    'grads' : [([0, 1, 2, 3], 0.), 
               ([1, 2, 3, 4], 0.)]
}
```

##### Requirements:

* rdkit
*numpy
