"""Optional standalone Keras composition and serialization checks."""
import numpy as np
import pytest

tf=pytest.importorskip("tensorflow")
from superonn.tensorflow import SuperONN1D, spectral_depthwise_conv1d, spectral_separable_conv1d


@pytest.mark.parametrize("precision",("stable","autocast"))
@pytest.mark.parametrize("jit",(False,True))
def test_keras_fit_and_serialization(tmp_path,precision,jit):
    tf.keras.utils.set_random_seed(37)
    layer=SuperONN1D(4,3,q=3,padding="same",input_bound=1,conv_precision=precision,jit_compile=jit)
    model=tf.keras.Sequential([tf.keras.Input((17,2)),layer])
    x=tf.random.stateless_uniform((2,17,2),(37,1))
    # Eager graph fit is independently useful on CPU-only release CI.
    model.compile(optimizer="adam",loss="mse",jit_compile=False)
    loss=model.train_on_batch(x,tf.zeros((2,17,4)))
    assert np.isfinite(loss)
    expected=model(x)
    path=tmp_path/"model.keras";model.save(path)
    restored=tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(x),expected,atol=1e-6)


def test_native_factory_shapes_and_gradients():
    x=tf.Variable(tf.ones((2,17,3))*.4)
    for layer in (spectral_depthwise_conv1d(3,operator="superonn"),
                  spectral_separable_conv1d(4,3,operator="superonn")):
        with tf.GradientTape() as tape:
            y=layer(x);loss=tf.reduce_mean(y*y)
        grads=tape.gradient(loss,[x]+layer.trainable_variables)
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
        assert tuple(y.shape[:2]) == (2,17)


def test_keras_spatial_tuple_syntax():
    x=tf.ones((1,17,3))*.4
    for layer in (SuperONN1D(4,(3,),strides=(2,),dilation_rate=(1,)),
                  spectral_depthwise_conv1d((3,),operator="superonn",strides=(2,)),
                  spectral_separable_conv1d(4,(3,),operator="superonn",dilation_rate=(2,))):
        y=layer(x)
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(y.shape)


def test_xla_symbolic_build_keeps_live_variables_and_shift_gradients():
    tf.keras.utils.set_random_seed(37)
    operator=SuperONN1D(2,3,q=3,groups=2,jit_compile=True)
    model=tf.keras.Sequential([tf.keras.Input((17,2)),operator])
    x=tf.reshape(tf.linspace(.1,.9,34),(1,17,2))
    with tf.GradientTape() as tape:
        before=model(x);loss=tf.reduce_sum(before*before)
    grads=tape.gradient(loss,operator.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
    index=next(i for i,v in enumerate(operator.trainable_variables) if v is operator.shift_parameter)
    assert float(tf.reduce_max(tf.abs(grads[index]))) > 0
    operator.kernel.assign(operator.kernel*2)
    np.testing.assert_allclose(model(x),before*2,atol=1e-6)
    config=operator.get_config();config['jit_compile']=False
    reference=SuperONN1D.from_config(config);reference(x);reference.set_weights(operator.get_weights())
    operator.shift_parameter.assign(tf.ones_like(operator.shift_parameter)*.2)
    reference.set_weights(operator.get_weights())
    np.testing.assert_allclose(model(x),reference(x),atol=1e-6)
