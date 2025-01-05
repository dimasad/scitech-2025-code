#!/usr/bin/env python3

"""Twin Otter lateral-directional motion, Variational System Identification.

Based on data and models (Textbook example 6.1) from SIDPAC
"Aircraft System Identification: Theory And Practice"
Second Edition
Authors: Morelli, Eugene A. and Klein, Vladislav
ISBN: 0-9974306-1-3

Data and reference model code available at
    https://software.nasa.gov/software/LAR-16100-1
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
import scipy.optimize
from visid import gvi, modeling, vi
from visid.benchmark import arggroups
from visid.modeling import (GaussianMeasurement, LinearTransitions,
                            GaussianTransition)


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
            data_dir / 'totter_f1_017_data.mat', 
            data_dir / 'totter_f1_014_data.mat'
        ]

    # Validate parameters
    assert all(f.exists() for f in args.datafiles), 'Datafiles missing.'

    # Return parsed arguments
    return args


class TOtterLat(GaussianTransition, GaussianMeasurement):
    """Twin Otter lateral-directional motion model."""

    nx: int = 4
    """Number of states."""

    nu: int = 2
    """Number of exogenous inputs."""

    ny: int = 5
    """Number of outputs."""

    dt: float = 0.02
    """Sampling period."""

    g: float = 32.174
    """Standard gravity."""

    Vo: float = 237.02
    """Trim airspeed."""

    a0: float = 0.0040
    """Trim angle of attack."""

    theta0: float = 0.0286
    """Trim pitch angle."""

    qbar: float = 56.3195
    """Dynamic pressure."""

    S: float = 422.4835
    """Wing area."""

    b: float = 64.9934
    """Wing span."""

    mass: float = 339.2963
    """Aircraft mass."""

    Ix: float = 20902
    """Moment of inertia about x-axis."""

    Iy: float = 24256
    """Moment of inertia about y-axis."""

    Iz: float = 38463
    """Moment of inertia about z-axis."""

    Ixz: float = 1127

    def setup(self):
        super().setup()
        self.CYb = self.param("CYb", nn.initializers.zeros, ())
        self.CYr = self.param("CYr", nn.initializers.zeros, ())
        self.CYdr = self.param("CYdr", nn.initializers.zeros, ())
        self.Clb = self.param("Clb", nn.initializers.zeros, ())
        self.Clp = self.param("Clp", nn.initializers.zeros, ())
        self.Clr = self.param("Clr", nn.initializers.zeros, ())
        self.Clda = self.param("Clda", nn.initializers.zeros, ())
        self.Cldr = self.param("Cldr", nn.initializers.zeros, ())
        self.Cnb = self.param("Cnb", nn.initializers.zeros, ())
        self.Cnp = self.param("Cnp", nn.initializers.zeros, ())
        self.Cnr = self.param("Cnr", nn.initializers.zeros, ())
        self.Cnda = self.param("Cnda", nn.initializers.zeros, ())
        self.Cndr = self.param("Cndr", nn.initializers.zeros, ())
        self.bias_bdot = self.param("bias_bdot", nn.initializers.zeros, ())
        self.bias_pdot = self.param("bias_pdot", nn.initializers.zeros, ())
        self.bias_rdot = self.param("bias_rdot", nn.initializers.zeros, ())
        self.bias_phidot = self.param("bias_phidot", nn.initializers.zeros, ())
        self.bias_ay = self.param("bias_ay", nn.initializers.zeros, ())

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        beta, p, r, phi = x
        da, dr = u

        # Unpack model parameters
        CYb = self.CYb
        CYr = self.CYr
        CYdr = self.CYdr
        Clb = self.Clb
        Clp = self.Clp
        Clr = self.Clr
        Clda = self.Clda
        Cldr = self.Cldr
        Cnb = self.Cnb
        Cnp = self.Cnp
        Cnr = self.Cnr
        Cnda = self.Cnda
        Cndr = self.Cndr
        bias_bdot = self.bias_bdot
        bias_pdot = self.bias_pdot
        bias_rdot = self.bias_rdot
        bias_phidot = self.bias_phidot
        bias_ay = self.bias_ay

        # Unpack constants and given parameters
        g = self.g
        Vo = self.Vo
        a0 = self.a0
        theta0 = self.theta0
        qbar = self.qbar
        S = self.S
        b = self.b
        mass = self.mass
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Ixz = self.Ixz

        # Compute secondary parameters and constants
        Gamma = Ix * Iz - Ixz**2
        c3 = Iz / Gamma
        c4 = Ixz / Gamma
        c9 = Ix / Gamma

        # Compute nondimensional rates
        phat = p * b / (2 * Vo)
        rhat = r * b / (2 * Vo)

        # Compute aerodynamic coefficients
        CY = CYb*beta + CYr*rhat + CYdr*dr
        Cl = Clb*beta + Clp*phat + Clr*rhat + Clda*da + Cldr*dr
        Cn = Cnb*beta + Cnp*phat + Cnr*rhat + Cnda*da + Cndr*dr

        # Compute forces and moments
        Y = qbar*S*CY
        L = qbar*S*b*Cl
        N = qbar*S*b*Cn

        # Compute state derivatives
        betadot = (Y/(mass*Vo) + p*jnp.sin(a0) - r*jnp.cos(a0) 
                   + g*jnp.cos(theta0)/Vo*phi + bias_bdot)
        pdot = c3*L + c4*N + bias_pdot
        rdot = c4*L + c9*N + bias_rdot
        phidot = p + jnp.tan(theta0)*r + bias_phidot

        # Assemble state derivative vector
        xdot = jnp.array([betadot, pdot, rdot, phidot])
        return xdot
    
    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        """State transition function."""
        xdot = self.fc(x, u)

        # Integrate to next step (Euler scheme)
        return x + xdot * self.dt

    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        """Output function."""
        # Unpack arguments
        beta, p, r, phi = x
        da, dr = u

        # Unpack model parameters
        CYb = self.CYb
        CYr = self.CYr
        CYdr = self.CYdr
        bias_ay = self.bias_ay

        # Unpack constants and given parameters
        g = self.g
        Vo = self.Vo
        qbar = self.qbar
        S = self.S
        b = self.b
        mass = self.mass

        # Compute nondimensional rates
        rhat = r * b / (2 * Vo)

        # Compute aerodynamic coefficients
        CY = CYb*beta + CYr*rhat + CYdr*dr

        # Compute forces
        Y = qbar*S*CY

        # ay reading
        ay = Y/(mass*g) + bias_ay
        
        # Output vector
        y = jnp.array([beta, p, r, phi, ay])
        return y


class Estimator(vi.VIBase):
    def setup(self):
        self.model = TOtterLat()
        self.smoother = gvi.SteadyStateSmoother(self.model.nx)
        self.sampler = gvi.SigmaPointSampler(self.model.nx)


if __name__ == '__main__':
    args = program_args()

    # Load Datafile
    d2r = np.pi / 180
    data = [None] * len(args.datafiles)
    for i, datafile in enumerate(args.datafiles):
        rawseg = scipy.io.loadmat(datafile.expanduser())['fdata']
        y = jnp.c_[
            d2r*rawseg[:, 2], d2r*rawseg[:, 4], d2r*rawseg[:, 6], 
            d2r*rawseg[:, 7], rawseg[:, 11]
        ]
        u = d2r * jnp.c_[rawseg[:, 14], rawseg[:, 15]]
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
