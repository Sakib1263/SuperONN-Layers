"""Additive independent branches with one optimizer update and regularizer.

Run after installation: python examples/tensorflow_branches.py
The Python loop schedules reusable graph calls; no global execution setting changes.
This recipe is for independent inputs and additive objectives, not coupled losses.
"""
import tensorflow as tf
from superonn.tensorflow import conv1d, model_regularization_loss


def branch_functions(model):
    """Create these functions once per built model, outside the step loop."""
    variables = tuple(model.trainable_variables)

    @tf.function(input_signature=(tf.TensorSpec((None, None, 1), tf.float32),
                                  tf.TensorSpec((None, 1), tf.float32)),
                 autograph=False, jit_compile=False)
    def branch_gradients(inputs, targets):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((model(inputs, training=True) - targets) ** 2)
        return loss, tape.gradient(loss, variables)

    @tf.function(autograph=False, jit_compile=False)
    def penalty_gradients():
        with tf.GradientTape() as tape:
            penalty = model_regularization_loss(model)
        gradients = tape.gradient(penalty, variables)
        return penalty, tuple(tf.zeros_like(v) if g is None else g
                              for v, g in zip(variables, gradients))

    return branch_gradients, penalty_gradients


def main():
    model = tf.keras.Sequential([
        tf.keras.Input((None, 1)),
        conv1d(3, 3, operator="superonn", padding="same", order_regularization=1e-5),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.SGD(.01)
    branch, regularizer = branch_functions(model)
    batches = [(tf.ones((2, size, 1)) * .2, tf.zeros((2, 1))) for size in (8, 10)]
    for step in range(2):
        pieces = [branch(x, y) for x, y in batches]
        penalty, penalty_grads = regularizer()
        # Equal mean branch weights; choose weights to match your objective.
        combined = [tf.add_n([gradients[i] for _, gradients in pieces]) / len(pieces) + extra
                    for i, extra in enumerate(penalty_grads)]
        optimizer.apply_gradients(zip(combined, model.trainable_variables))
        loss = tf.add_n([value for value, _ in pieces]) / len(pieces) + penalty
        print(f"step={step + 1} loss={float(loss):.6f}")
    print(f"branch_traces={branch.experimental_get_tracing_count()}")


if __name__ == "__main__":
    main()
