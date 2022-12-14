import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at
from rdkit import Chem
from conf_calc import ConfCalc
import matplotlib.pyplot as plt
import xarray as xr


def get_prepared_calculator(mol_file):
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    calculator = ConfCalc(mol=mol,
                          dir_to_xyzs="xtb_calcs/",
                          rotable_dihedral_idxs=[[5, 4, 6, 7],
                                                 [4, 6, 7, 9],
                                                 [12, 13, 15, 16]])

    zero_level = calculator.log_prob(np.array([0.0, 0.0, 0.0]))
    calculator.norm_en = zero_level

    return calculator


class LogLikeGrad(at.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, log_prob_grad):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.log_prob_grad = log_prob_grad

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = self.log_prob_grad(theta)

        outputs[0][0] = grads


class LogLike(at.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, log_prob, log_prob_grad):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        log_prob:
            The log of probability
        log_prob_grad:
            The gradient of log of probability
        """

        # add inputs as class attributes
        self.log_prob = log_prob
        self.log_prob_grad = LogLikeGrad(log_prob_grad)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.log_prob(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.log_prob_grad(theta)]


def from_trace(trace):
    xs_4_chanks = trace.posterior.pot
    xs = xr.concat(xs_4_chanks, dim='draw')

    ys_4_chanks = trace.sample_stats.lp
    ys = xr.concat(ys_4_chanks, dim='draw')

    return xs, ys


def to_csv(Xs, Ys, filename='./data.csv'):
    names = list()
    data = list()
    for x in range(Xs.pot_dim_0.shape[0]):
        names.append(f'x.{x+1}')
        data.append(Xs.isel(pot_dim_0=x))
    names.append('lp')
    data.append(Ys)
    pd.DataFrame.from_dict(dict(zip(names, data))).to_csv(filename)


# def bad(values: np.ndarray) -> float:
#     return -(values[0]*values[0] + values[1]*values[1]+ values[2]*values[2])
#
#
# def bad_grad(values: np.ndarray):
#     return np.array([-2*values[0], -2*values[1], -2*values[2]])


def main():
    calc = get_prepared_calculator("test.mol")

    logl = LogLike(calc.log_prob, calc.log_prob_grad)
    # logl = LogLike(bad, bad_grad)

    with pm.Model() as model:
        pm.DensityDist(
            "pot",
            logp=logl,
            shape=3,
        )

    sample_num = 10
    with model:
        trace = pm.sample(sample_num)

    xs, ys = from_trace(trace)
    to_csv(xs, ys)


if __name__ == '__main__':
    main()