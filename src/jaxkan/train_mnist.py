import jax
import jax.numpy as jnp
from jaxkan.model import model
import matplotlib.pyplot as plt
import optax
from jaxkan.mnist_load import get_dataset_torch

class_num = 10
sample_per_class = 1000
input_dim = 28 * 28
batch_size = 64
epoch_num = 10
lr = 0.003


basis_fn = jax.nn.silu
width_list = [input_dim, 64, class_num]
grid_size = 5  # fine-grainedness of grid. more accurate when larger
k = 3  # order of spline
grid_range = [-1, 1]
t = jnp.arange(grid_range[0], grid_range[1], 1 / grid_size)

# each psi(x_i) needs parameters of basis coef(length: len(t)-k-1) + scale_base(length: 1) + scale_spline(length: 1)
psi_param_length = len(t) - k - 1 + 2
param_size = sum(
    [
        width_list[l] * width_list[l + 1] * psi_param_length
        for l in range(len(width_list) - 1)
    ]
)
print("param_size", param_size)
params = (
    jax.random.normal(jax.random.PRNGKey(0), shape=(param_size,), dtype=jnp.float32)
    * 0.1
)

train_ds, test_ds = get_dataset_torch(class_num, sample_per_class)
print("data loaded")


solver = optax.adam(learning_rate=lr)
opt_state = solver.init(params)


def loss_fn(params, X, Y):
    logits = jax.vmap(
        lambda x: jax.nn.log_softmax(model(params, x, basis_fn, width_list, t, k))
    )(X)
    one_hots = jax.nn.one_hot(Y, class_num)
    one_hots = jnp.reshape(one_hots, (len(Y), class_num))

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hots))
    return loss, logits


train_ds_size = len(train_ds["image"])
steps_per_epoch = train_ds_size // batch_size

loss_history = []
train_accuracy_history = []
test_accuracy_history = []

keys = jax.random.split(jax.random.PRNGKey(0), epoch_num)
for epoch in range(epoch_num):
    perms = jax.random.permutation(keys[epoch], len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    for perm in perms:
        batch_images = train_ds["image"][perm, ...].reshape((batch_size, input_dim))
        batch_labels = train_ds["label"][perm, ...].reshape((batch_size, 1))
        (loss, logits), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_images, batch_labels
        )
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(loss)
    train_accuracy = jnp.mean(
        jax.vmap(
            lambda x, y: jnp.argmax(model(params, x, basis_fn, width_list, t, k)) == y
        )(
            train_ds["image"].reshape((-1, input_dim)),
            train_ds["label"].reshape((-1, 1)),
        )
    )
    test_accuracy = jnp.mean(
        jax.vmap(
            lambda x, y: jnp.argmax(model(params, x, basis_fn, width_list, t, k)) == y
        )(test_ds["image"].reshape((-1, input_dim)), test_ds["label"].reshape((-1, 1)))
    )
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)

    print(
        f"epoch {epoch} loss: {loss:.3f} train_accuracy: {train_accuracy:.3f} test_accuracy: {test_accuracy:.3f}"
    )


plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("mnist_loss.png")


plt.figure()
plt.plot(train_accuracy_history, label="train")
plt.plot(test_accuracy_history, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("mnist_accuracy.png")
