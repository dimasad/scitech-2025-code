#!/usr/bin/env python3

"""HFB-320 longitudinal motion (Cl, CD, Cm) Variational System Identification.

Based on data and code (test case 04) from 
"Flight Vehicle System Identification - A Time Domain Methodology"
Second Edition
Author: Ravindra V. Jategaonkar
Published by AIAA, Reston, VA 20191, USA

Data available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip

Code available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/chapter05.zip
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
        '--maxiter', type=int, default=1000,
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
        args.datafiles = [data_dir / 'hfb320_1_10.asc']

    # Validate parameters
    assert all(f.exists() for f in args.datafiles), 'Datafiles missing.'

    # Return parsed arguments
    return args


class HFB320(GaussianTransition, GaussianMeasurement, sde.Euler):

    nx: int = 4
    """Number of states."""

    nu: int = 2
    """Number of exogenous inputs."""

    ny: int = 7
    """Number of outputs."""

    dt: float = 0.1
    """Sampling period."""

    def setup(self):
        super().setup()

        self.CD0 = self.param("CD0", nn.initializers.zeros, ())
        self.CDV = self.param("CDV", nn.initializers.zeros, ())
        self.CDAL = self.param("CDAL", nn.initializers.zeros, ())
        self.CL0 = self.param("CL0", nn.initializers.zeros, ())
        self.CLV = self.param("CLV", nn.initializers.zeros, ())
        self.CLAL = self.param("CLAL", nn.initializers.zeros, ())
        self.CM0 = self.param("CM0", nn.initializers.zeros, ())
        self.CMV = self.param("CMV", nn.initializers.zeros, ())
        self.CMAL= self.param("CMAL", nn.initializers.zeros, ())
        self.CMQ = self.param("CMQ", nn.initializers.zeros, ())
        self.CMDE = self.param("CMDE", nn.initializers.zeros, ())
    
    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        VT, Alfa, The, Qrate = x
        de, Fe = u

        # Unpack model parameters
        CD0 = self.CD0
        CDV = self.CDV
        CDAL = self.CDAL
        CL0 = self.CL0
        CLV = self.CLV
        CLAL = self.CLAL
        CM0 = self.CM0
        CMV = self.CMV
        CMAL = self.CMAL
        CMQ = self.CMQ
        CMDE = self.CMDE

        # Model constants
        G0     =    9.80665e0
        SBYM   =    4.02800e-3
        SCBYIY =    8.00270e-4
        FEIYLT =   -7.01530e-6
        V0     =  104.67000e0
        RM     = 7472.00000e0
        SIGMAT =    0.05240e0
        RHO    =    0.79200e0

        #Intermediate variables
        QBAR   = 0.5 * RHO *VT**2                  

        from jax.numpy import cos, sin

        # Right sides of state equations (5.86)
        VTdot = -SBYM*QBAR    * (CD0 + CDV*VT/V0 + CDAL*Alfa) \
                                    + Fe*cos(Alfa + SIGMAT)/RM + G0*sin(Alfa - The)

        Aldot = -SBYM*QBAR/VT * (CL0 + CLV*VT/V0 + CLAL*Alfa ) \
                                    - Fe*sin(Alfa + SIGMAT)/(RM*VT) + Qrate + G0*cos(Alfa - The)/VT

        Qdot  =  SCBYIY*QBAR  * (CM0 + CMV*VT/V0 + CMAL*Alfa \
                                    + 1.215*CMQ*Qrate/V0 + CMDE*de ) + FEIYLT*Fe

        # Assemble state derivative vector
        xdot = jnp.array([VTdot, Aldot, Qrate, Qdot])
        return xdot
    
    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        """Output function."""
        # Unpack arguments
        VT, Alfa, The, Qrate = x
        de, Fe = u

        # Unpack model parameters
        CD0 = self.CD0
        CDV = self.CDV
        CDAL = self.CDAL
        CL0 = self.CL0
        CLV = self.CLV
        CLAL = self.CLAL
        CM0 = self.CM0
        CMV = self.CMV
        CMAL = self.CMAL
        CMQ = self.CMQ
        CMDE = self.CMDE

        # Model constants
        G0     =    9.80665e0
        SBYM   =    4.02800e-3
        SCBYIY =    8.00270e-4
        FEIYLT =   -7.01530e-6
        V0     =  104.67000e0
        RM     = 7472.00000e0
        SIGMAT =    0.05240e0
        RHO    =    0.79200e0
        cbarH  =    1.215

        #Intermediate variables
        QBAR   = 0.5 * RHO *VT**2                  

        from jax.numpy import cos, sin

        CD    =  CD0 + CDV*VT/V0 + CDAL*Alfa;
        CL    =  CL0 + CLV*VT/V0 + CLAL*Alfa;
        SALFA =  sin(Alfa);
        CALFA =  cos(Alfa);
        CX    =  CL*SALFA - CD*CALFA;
        CZ    = -CL*CALFA - CD*SALFA;

        # Output vector
        y = jnp.array([VT,
            Alfa,
            The,
            Qrate,
            SCBYIY*QBAR * (CM0 + CMV*VT/V0 + CMAL*Alfa + CMQ*Qrate*cbarH/V0 + CMDE*de) \
            + FEIYLT*Fe,
            QBAR*SBYM*CX + Fe*cos(SIGMAT)/RM,
            QBAR*SBYM*CZ - Fe*sin(SIGMAT)/RM])
        return y


class Estimator(vi.VIBase):
    def setup(self):
        self.model = HFB320()
        self.smoother = gvi.SteadyStateSmoother(self.model.nx)
        self.sampler = gvi.SigmaPointSampler(self.model.nx)


if __name__ == '__main__':
    args = program_args()

    # Load Datafile
    rawdata = [np.loadtxt(f.expanduser()) for f in args.datafiles]
    data = [None] * len(rawdata)
    for i, rawseg in enumerate(rawdata):
        y = jnp.c_[
            rawseg[:, 4], rawseg[:, 5], rawseg[:, 6], rawseg[:, 7], 
            rawseg[:, 8], rawseg[:, 9], rawseg[:, 10]
        ]
        u = jnp.c_[rawseg[:, 1], rawseg[:, 3]]
        data[i] = vi.Data(y, u)
    dataest = data
    dataval = None

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
