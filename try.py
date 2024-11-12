import jax
import netket as nk
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import json
import matplotlib.pyplot as plt

# 1D Lattice
L = 16
gp = 1
Vp = 2.0

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

# Observables
obs_sx = nk.operator.LocalOperator(hi)
obs_sy = nk.operator.LocalOperator(hi, dtype=complex)
obs_sz = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += (gp / 2.0) * nk.operator.spin.sigmax(hi, i)
    ha += (
        (Vp / 4.0)
        * nk.operator.spin.sigmaz(hi, i)
        * nk.operator.spin.sigmaz(hi, (i + 1) % L)
    )
    # sigma_{-} dissipation on every site
    j_ops.append(nk.operator.spin.sigmam(hi, i))
    obs_sx += nk.operator.spin.sigmax(hi, i)
    obs_sy += nk.operator.spin.sigmay(hi, i)
    obs_sz += nk.operator.spin.sigmaz(hi, i)


#  Create the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

ma = nk.models.NDM(
    beta=1,
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(lind.hilbert)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

vs = nk.vqs.MCMixedState(sa, ma, n_samples=2000, n_samples_diag=512)
vs.init_parameters(jax.nn.initializers.normal(stddev=0.01))

ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

obs = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}

out = ss.run(n_iter=300, out="test", obs=obs)
