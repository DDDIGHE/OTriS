import flax.linen as nn
import jax.numpy as jnp

class JasShort(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Define the two variational parameters J1 and J2
        j1 = self.param(
            "j1", nn.initializers.normal(), (1,), float
        )
        j2 =self.param(
            "j2", nn.initializers.normal(), (1,), float
        )
        # compute the nearest-neighbor correlations
        corr1=x*jnp.roll(x,-1,axis=-1)
        corr2=x*jnp.roll(x,-2,axis=-1)
        # sum the output
        return jnp.sum(j1*corr1+j2*corr2,axis=-1)