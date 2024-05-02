import jax
import jax.numpy as jnp
from jaxkan.model import model

dataset_size = 100
input_dim = 2


@jax.jit
def f(x):
    # return x[0] * 2 + x[1] * 3
    return jnp.exp(jnp.sin(jnp.pi * x[0]) + x[1] ** 2)
    # return jnp.exp(jnp.sin(x[0]**2 + x[1]**2) + jnp.sin(x[2]**2 + x[3]**2))


X = jax.random.uniform(
    jax.random.PRNGKey(0),
    shape=(dataset_size, input_dim),
    dtype=jnp.float32,
    minval=-1.0,
    maxval=1.0,
)
# normalize
Y = jnp.array([f(x) for x in X])

basis_fn = jax.nn.silu
width_list = [2, 1, 1]
num_grid_interval = 3
t = jnp.linspace(-1, 1, num_grid_interval + 1)
k = 3
coef_length = len(t) - k - 1 + 1
param_size = sum(
    [
        width_list[l] * width_list[l + 1] * coef_length
        for l in range(len(width_list) - 1)
    ]
)
coef = (
    jax.random.normal(jax.random.PRNGKey(0), shape=(param_size,), dtype=jnp.float32)
    * 0.1
)


def loss_fn(coef, x, y):
    predict = model(coef, x, basis_fn, width_list, t, k)
    return jnp.mean((predict - y) ** 2)


def batched_loss_fn(coef, X, Y):
    return jnp.mean(jax.vmap(lambda x, y: loss_fn(coef, x, y))(X, Y))


for i in range(1000):
    val, grad = jax.value_and_grad(batched_loss_fn)(coef, X, Y)
    coef = coef - 0.1 * grad
    if i % 10 == 0:
        print(f"(step {i}) loss: {val}")


print("loss:", batched_loss_fn(coef, X, Y))
for x, y in zip(X, Y):
    print("x:", x, "y:", y, "predict:", model(coef, x, basis_fn, width_list, t, k))
