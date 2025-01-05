#!/usr/bin/env python3

"""ATTAS lateral-directional motion, ny=5, Variational System Identification.

Based on data and code (test case 01) from 
"Flight Vehicle System Identification - A Time Domain Methodology"
Second Edition
Author: Ravindra V. Jategaonkar
Published by AIAA, Reston, VA 20191, USA

Data available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip

Code available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/chapter04.zip    
"""


import argparse
import pathlib
import sys

import flax.linen as nn
import hedeut as utils
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
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
            data_dir / 'fAttasAil1_pqrDot.asc',
            data_dir / 'fAttasRud1_pqrDot.asc',
            data_dir / 'fAttasAilRud2.asc'
        ]

    # Validate parameters
    assert all(f.exists() for f in args.datafiles), 'Datafiles missing.'

    # Return parsed arguments
    return args


class AttasLat(GaussianTransition, GaussianMeasurement, sde.Euler):
    """ATTAS lateral-directional motion model with 5 outputs."""

    nx: int = 2
    """Number of states."""

    nu: int = 3
    """Number of exogenous inputs."""

    ny: int = 5
    """Number of outputs."""

    dt: float = 0.04
    """Sampling period."""

    def setup(self):
        super().setup()

        self.Lp = self.param("Lp", nn.initializers.zeros, ())
        self.Lr = self.param("Lr", nn.initializers.zeros, ())
        self.Lda = self.param("Lda", nn.initializers.zeros, ())
        self.Ldr = self.param("Ldr", nn.initializers.zeros, ())
        self.Lbeta = self.param("Lbeta", nn.initializers.zeros, ())
        self.Np = self.param("Np", nn.initializers.zeros, ())
        self.Nr = self.param("Nr", nn.initializers.zeros, ())
        self.Nda = self.param("Nda", nn.initializers.zeros, ())
        self.Ndr = self.param("Ndr", nn.initializers.zeros, ())
        self.Nbeta = self.param("Nbeta", nn.initializers.zeros, ())
        self.Yp = self.param("Yp", nn.initializers.zeros, ())
        self.Yr = self.param("Yr", nn.initializers.zeros, ())
        self.Yda = self.param("Yda", nn.initializers.zeros, ())
        self.Ydr = self.param("Ydr", nn.initializers.zeros, ())
        self.Ybeta = self.param("Ybeta", nn.initializers.zeros, ())
        self.BX1 = self.param("BX1", nn.initializers.zeros, ())
        self.BX2 = self.param("BX2", nn.initializers.zeros, ())
        self.BY1 = self.param("BY1", nn.initializers.zeros, ())
        self.BY2 = self.param("BY2", nn.initializers.zeros, ())
        self.BY3 = self.param("BY3", nn.initializers.zeros, ())
        self.BY4 = self.param("BY4", nn.initializers.zeros, ())
        self.BY5 = self.param("BY5", nn.initializers.zeros, ())

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        p, r = x
        dela, delr, beta = u

        # Unpack model parameters
        Lp = self.Lp
        Lr = self.Lr
        Lda = self.Lda
        Ldr = self.Ldr
        Lbeta = self.Lbeta
        Np = self.Np
        Nr = self.Nr
        Nda = self.Nda
        Ndr = self.Ndr
        Nbeta = self.Nbeta
        Yp = self.Yp
        Yr = self.Yr
        Yda = self.Yda
        Ydr = self.Ydr
        Ybeta = self.Ybeta
        BX1 = self.BX1
        BX2 = self.BX2
        BY1 = self.BY1
        BY2 = self.BY2
        BY3 = self.BY3
        BY4 = self.BY4
        BY5 = self.BY5

        # Compute state derivatives
        pdot = Lp*p + Lr*r + Lda*dela + Ldr*delr + Lbeta*beta + BX1
        rdot = Np*p + Nr*r + Nda*dela + Ndr*delr + Nbeta*beta + BX2

        # Assemble state derivative vector
        xdot = jnp.array([pdot, rdot])
        return xdot
    
    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        """Output function."""
        # Unpack arguments
        p, r = x
        dela, delr, beta = u

        # Unpack model parameters
        Lp = self.Lp
        Lr = self.Lr
        Lda = self.Lda
        Ldr = self.Ldr
        Lbeta = self.Lbeta
        Np = self.Np
        Nr = self.Nr
        Nda = self.Nda
        Ndr = self.Ndr
        Nbeta = self.Nbeta
        Yp = self.Yp
        Yr = self.Yr
        Yda = self.Yda
        Ydr = self.Ydr
        Ybeta = self.Ybeta
        BX1 = self.BX1
        BX2 = self.BX2
        BY1 = self.BY1
        BY2 = self.BY2
        BY3 = self.BY3
        BY4 = self.BY4
        BY5 = self.BY5

        # Compute state derivatives
        pdot = Lp*p + Lr*r + Lda*dela + Ldr*delr + Lbeta*beta + BX1
        rdot = Np*p + Nr*r + Nda*dela + Ndr*delr + Nbeta*beta + BX2
        ay   = Yp*p + Yr*r + Yda*dela + Ydr*delr + Ybeta*beta

        # Output vector
        y = jnp.array([pdot + BY1, rdot + BY2, ay + BY3, p + BY4, r + BY5])
        return y


class Estimator(vi.VIBase):
    def setup(self):
        self.model = AttasLat()
        self.smoother = gvi.SteadyStateSmoother(self.model.nx)
        self.sampler = gvi.SigmaPointSampler(self.model.nx)


if __name__ == '__main__':
    args = program_args()

    # Load Datafile
    d2r = np.pi / 180
    rawdata = [np.loadtxt(f.expanduser()) for f in args.datafiles]
    data = [None] * len(rawdata)
    for i, rawseg in enumerate(rawdata):
        y = jnp.c_[
            rawseg[:, 16]*d2r, rawseg[:, 18]*d2r, rawseg[:, 2]*d2r,
            rawseg[:, 6]*d2r, rawseg[:, 8]*d2r
        ]
        u = jnp.c_[(rawseg[:, 28]-rawseg[:, 27])*d2r/2, rawseg[:, 29]*d2r,
                    rawseg[:, 13]*d2r]
        data[i] = vi.Data(y, u)
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
