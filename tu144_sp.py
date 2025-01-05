#!/usr/bin/env python3

"""Twin Otter short-period motion, Variational System Identification.

Based on data and models (Textbook example 6.3) from SIDPAC
"Aircraft System Identification: Theory And Practice"
Second Edition
Authors: Morelli, Eugene A. and Klein, Vladislav
ISBN: 0-9974306-1-3

Data and reference model code available at
    https://software.nasa.gov/software/LAR-16100-1
"""


import argparse
import dataclasses
import pathlib
import sys

import flax.linen as nn
import hedeut as utils
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from visid import gvi, modeling, sde, vi
from visid.benchmark import arggroups
from visid.modeling import (GaussianMeasurement, GaussianTransition,
                            LinearTransitions)


def program_args():
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        '--datafiles', type=pathlib.Path, nargs='*', help='Input data files.'
    )
    parser.add_argument(
        '--maxiter', type=int, default=200,
        help='Maximum number of optimizer iterations.'
    )
    parser.add_argument(
        '--output', type=pathlib.Path,
        help='File name base for saving the script output.'
    )
    parser.add_argument(
        '--plot', action='store_true', help='Show plots interactively.'
    )

    # Add common benchmark argument groups
    arggroups.add_jax_group(parser, jax_x64=True, jax_platform='cpu')
    arggroups.add_testing_group(parser)

    # Parse command-line arguments
    args = parser.parse_args()

    # Process common benchmark argument groups
    arggroups.process(args)

    # Choose default datafiles if none selected
    if not args.output:
        script_path = pathlib.Path(sys.argv[0])
        out_dir = script_path.parent / 'output'
        args.output = out_dir / script_path.with_suffix('.plot').name

    # Choose default datafiles if none selected
    if not args.datafiles:
        data_dir = pathlib.Path(sys.argv[0]).parent / 'data'
        args.datafiles = [
            data_dir / 'tu144_f20_4d_data.mat',
            data_dir / 'tu144_f20_4f_pred_data.mat',
        ]

    # Validate parameters
    assert all(f.exists() for f in args.datafiles), 'Datafiles missing.'

    # Return parsed arguments
    return args


class Tu144SP(GaussianTransition, GaussianMeasurement, sde.Euler):
    """Tupolev Tu-144LL short-period motion model."""

    nx: int = 2
    """Number of states."""

    nu: int = 1
    """Number of exogenous inputs."""

    ny: int = 3
    """Number of outputs."""

    dt: float = 0.03125
    """Sampling period."""

    g: float = 32.17405
    """Standard gravity."""

    Vo: float = 865.9
    """Trim airspeed."""

    def setup(self):
        super().setup()

        self.Z0 = self.param("Z0", nn.initializers.zeros, ())
        self.Za = self.param("Za", nn.initializers.zeros, ())
        self.Zq = self.param("Zq", nn.initializers.zeros, ())
        self.Zde = self.param("Zde", nn.initializers.zeros, ())
        self.M0 = self.param("M0", nn.initializers.zeros, ())
        self.Ma = self.param("Ma", nn.initializers.zeros, ())
        self.Mq = self.param("Mq", nn.initializers.zeros, ())
        self.Mde = self.param("Mde", nn.initializers.zeros, ())
        self.az0 = self.param("az0", nn.initializers.zeros, ())

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        a, q = x
        de, = u

        # Unpack model parameters
        Z0 = self.Z0
        Za = self.Za
        Zq = self.Zq
        Zde = self.Zde
        M0 = self.M0
        Ma = self.Ma
        Mq = self.Mq
        Mde = self.Mde

        # Compute state derivatives
        adot = Z0 + Za*a + (Zq + 1)*q + Zde*de
        qdot = M0 + Ma*a +       Mq*q + Mde*de

        # Assemble state derivative vector
        xdot = jnp.array([adot, qdot])
        return xdot
    
    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        """Output function."""
        # Unpack arguments
        a, q = x
        de, = u

        # Unpack constants and given parameters
        g = self.g
        Vo = self.Vo

        # Unpack model parameters
        Za = self.Za
        Zq = self.Zq        
        Zde = self.Zde
        az0 = self.az0

        # Compute az reading
        az = Vo / g * (Za*a + Zq*q + Zde*de) + az0

        # Assemble output vector and return
        y = jnp.array([a, q, az])
        return y


class Estimator(vi.VIBase):
    model_args: dict = dataclasses.field(default_factory=dict)
    """Initialization arguments of underlying dynamical system model."""

    def setup(self):
        self.model = Tu144SP(**self.model_args)
        self.smoother = gvi.SteadyStateSmoother(self.model.nx)
        self.sampler = gvi.SigmaPointSampler(self.model.nx)


if __name__ == '__main__':
    args = program_args()

    # Load Datafiles (all segments)
    d2r = np.pi / 180
    data = [None] * len(args.datafiles)
    for i, datafile in enumerate(args.datafiles):
        rawseg = scipy.io.loadmat(datafile.expanduser())['fds']['fdata'][0, 0]
        y = jnp.c_[rawseg[:, 3]*d2r, rawseg[:, 5]*d2r, rawseg[:, 12]]
        u = jnp.c_[rawseg[:, 13]*d2r]
        data[i] = vi.Data(y, u)
    # Split data into estimation and validation sets
    dataest = data[:-1]
    dataval = data[-1]

    # Create the PRNG keys
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)

    # Instantiate and initialize the estimator
    estimator = Estimator()
    v = vi.multiseg_init(estimator, dataest, init_key)

    # Run optimization
    optimizer = vi.Optimizer(estimator, v, dataest, multiseg=True)
    v, sol = optimizer(
        method='trust-constr', options={'maxiter': args.maxiter, 'verbose': 2}
    )

    # Get estimated model and its parameters
    model = estimator.bind(v[0]).model
    model_params = v[0]['params']['model']
    xsmooth = [vseg['params']['smoother']['mu'] for vseg in v]

    # Get model responses for estimation data
    res_est = modeling.compare(model, xsmooth, dataest)

    # Save estimation results
    if args.output:
        out = args.output
        t = [0]
        for i, results in enumerate(zip(dataest, *res_est)):
            dataseg, ys, ysim, ypred, xdot, f = results
            t = np.arange(len(dataseg.y)) * model.dt + t[-1]
            np.savetxt(out.with_suffix(f'.{i}.y.plot'), jnp.c_[t, dataseg.y])
            np.savetxt(out.with_suffix(f'.{i}.u.plot'), jnp.c_[t, dataseg.u])
            np.savetxt(out.with_suffix(f'.{i}.ys.plot'), jnp.c_[t, ys])
            np.savetxt(out.with_suffix(f'.{i}.ysim.plot'), jnp.c_[t, ysim])
            np.savetxt(out.with_suffix(f'.{i}.ypred.plot'), jnp.c_[t, ypred])
            np.savetxt(out.with_suffix(f'.{i}.xdot.plot'), jnp.c_[t[:-1], xdot])
            np.savetxt(out.with_suffix(f'.{i}.fc.plot'), jnp.c_[t[:-1], f])

    # Plot estimation results on screen
    if args.plot:
        from matplotlib import pyplot as plt
        t = [0]
        for i, results in enumerate(zip(dataest, *res_est)):
            dataseg, ys, ysim, ypred, xdot, f = results
            t = np.arange(len(dataseg.y)) * model.dt + t[-1]
            for j in range(model.ny):
                plt.figure(j)
                plt.plot(t, dataseg.y[:,j], '.')
                plt.plot(t, ys[:,j], '-')                
                plt.plot(t, ysim[:,j], '--')
                plt.plot(t, ypred[:,j], ':')
                plt.xlabel('Time [s]')
                plt.ylabel(f'Output {j}')
                plt.title(f'Estimation results for output {j}')
            for j in range(model.nx):
                plt.figure(j + model.ny)
                plt.plot(t[:-1], xdot[:,j], '.')
                plt.plot(t[:-1], f[:,j], '-')
                plt.xlabel('Time [s]')
                plt.ylabel(f'xdot {j}')
                plt.title(f'Estimation results for state {j}')

    # Return if no validation data is available
    if dataval is None:
        raise SystemExit

    # Run the smoother on the validation dataset
    vval = vi.fixed_model_init(estimator, dataval, init_key)
    valsmoother = vi.Optimizer(
        estimator, vval, dataval, model_params=model_params
    )
    vval, solval = valsmoother(
        method='trust-constr', options={'maxiter': args.maxiter, 'verbose': 2}
    )
    xval = vval['params']['smoother']['mu']

    # Get model responses for validation data
    res_val = modeling.compare(model, [xval], [dataval])

    # Save validation results
    if args.output:
        out = args.output
        (ys, ysim, ypred, xdot, f), = zip(*res_val)
        t = np.arange(len(dataval.y)) * model.dt
        np.savetxt(out.with_suffix(f'.val.y.plot'), jnp.c_[t, dataval.y])
        np.savetxt(out.with_suffix(f'.val.u.plot'), jnp.c_[t, dataval.u])
        np.savetxt(out.with_suffix(f'.val.ys.plot'), jnp.c_[t, ys])
        np.savetxt(out.with_suffix(f'.val.ysim.plot'), jnp.c_[t, ysim])
        np.savetxt(out.with_suffix(f'.val.ypred.plot'), jnp.c_[t, ypred])
        np.savetxt(out.with_suffix(f'.val.xdot.plot'), jnp.c_[t[:-1], xdot])
        np.savetxt(out.with_suffix(f'.val.fc.plot'), jnp.c_[t[:-1], f])

    # Plot validation results on screen
    if args.plot:
        from matplotlib import pyplot as plt
        (ys, ysim, ypred, xdot, f), = zip(*res_val)
        t = np.arange(len(dataval.y)) * model.dt
        for j in range(model.ny):
            plt.figure(j + model.ny + model.nx)
            plt.plot(t, dataval.y[:,j], '.')
            plt.plot(t, ys[:,j], '-')                
            plt.plot(t, ysim[:,j], '--')
            plt.plot(t, ypred[:,j], ':')
            plt.xlabel('Time [s]')
            plt.ylabel(f'Output {j}')
            plt.title(f'Validation results for output {j}')
        for j in range(model.nx):
            plt.figure(j + 2*model.ny + model.nx)
            plt.plot(t[:-1], xdot[:,j], '.')
            plt.plot(t[:-1], f[:,j], '-')
            plt.xlabel('Time [s]')
            plt.ylabel(f'xdot {j}')
            plt.title(f'Validation results for state {j}')
