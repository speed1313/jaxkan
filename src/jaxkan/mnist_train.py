import jax
import jax.numpy as jnp
from jaxkan.model import model
import matplotlib.pyplot as plt
import optax
from jaxkan.dataset import get_dataset_torch

class_num = 2
dataset_size = 100
input_dim = 28 * 28
batch_size = 100
epoch_num = 50
lr = 0.01


basis_fn = jax.nn.silu
width_list = [input_dim, 10, class_num]
grid_size = 10
k = 3
grid_range = [-1, 1]
t = jnp.arange(grid_range[0], grid_range[1], 1/grid_size)
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

train_ds, test_ds = get_dataset_torch(class_num, dataset_size)
print("data loaded")
print("input_range", jnp.min(train_ds["image"]), jnp.max(train_ds["image"]))


def loss_fn(coef, X, Y):
    logits = jax.vmap(lambda x: model(coef, x, basis_fn, width_list, t, k))(X)
    one_hots = jax.nn.one_hot(Y, class_num)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hots))
    return loss, logits


train_ds_size = len(train_ds["image"])
steps_per_epoch = train_ds_size // batch_size

loss_history = []
accuracy_history = []
for epoch in range(epoch_num):
    perms = jax.random.permutation(jax.random.key(epoch), len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_images = jnp.reshape(batch_images, (batch_size, input_dim))
        batch_labels = train_ds["label"][perm, ...]
        batch_labels = jnp.reshape(batch_labels, (batch_size, 1))
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(coef, batch_images, batch_labels)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        coef = coef - lr * grads
    print(f"epoch {epoch} loss: {loss}, accuracy: {accuracy}")


plt.plot(loss_history)
# log scale
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("mnist_loss.png")
plt.show()
plt.plot(accuracy_history)
plt.xlabel("step")
plt.ylabel("accuracy")
plt.savefig("mnist_accuracy.png")

# Accuracy on test set
test_images = jnp.reshape(test_ds["image"], (len(test_ds["image"]), input_dim))
test_labels = jnp.reshape(test_ds["label"], (len(test_ds["label"]), 1))
accuracy = jnp.mean(jax.vmap(lambda x, y: jnp.argmax(model(coef, x, basis_fn, width_list, t, k)) == y)(test_images, test_labels))
print(f"accuracy: {accuracy}")





