import pymc as pm
import numpy as np
import xarray as xr
import pandas as pd
import aesara.tensor as at


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

        self.log_prob_grad = log_prob_grad

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = self.log_prob_grad(theta)
        # print(f'{theta=} {grads=}')
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
        # print(f'{theta=} {logl=}')

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        grad = self.log_prob_grad(theta)
        return [g[0] * grad]


def from_trace(trace, log_prob):
    xs_4_chanks = trace.posterior.theta
    xs = xr.concat(xs_4_chanks, dim='draw')

    ys = np.apply_along_axis(log_prob, 1, xs)
    return xs, ys


def to_csv(xs, ys, filename='./data_one.csv'):
    names = list()
    data = list()
    for x in range(xs.theta_dim_0.shape[0]):
        names.append(f'theta.{x+1}')
        data.append(xs.isel(theta_dim_0=x))
    names.append('lp')
    data.append(ys)
    pd.DataFrame.from_dict(dict(zip(names, data))).to_csv(filename)


# def bad(values: np.ndarray) -> float:
#     return -(values[0]*values[0] + values[1]*values[1]+ values[2]*values[2])
#
#
# def bad_grad(values: np.ndarray):
#     return np.array([-2*values[0], -2*values[1], -2*values[2]])


def main(calc, sample_num=1000):
    logl = LogLike(calc.log_prob_one, calc.log_prob_grad_one)

    with pm.Model() as model:
        interval = pm.distributions.transforms.Interval(lower=-np.pi, upper=np.pi)
        pm.DensityDist(
            "theta",
            logp=logl,
            shape=1,
            transform=interval
        )

    with model:
        trace = pm.sample(sample_num, tune=sample_num)

    def log_prob(x):
        energy, _ = calc.get_energy_one(x)
        return energy

    xs, ys = from_trace(trace, log_prob)
    to_csv(xs, ys)
