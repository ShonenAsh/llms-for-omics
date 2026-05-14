============================= test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0 -- /usr/local/bin/python3.13
cachedir: .pytest_cache
rootdir: /workspace
plugins: anyio-4.13.0
collecting ... collected 63 items

tests/test_01_tensor_basics.py::test_1a_ones_times_five PASSED           [  1%]
tests/test_01_tensor_basics.py::test_1b_matmul PASSED                    [  3%]
tests/test_01_tensor_basics.py::test_1c_reduce_last_dim PASSED           [  4%]
tests/test_01_tensor_basics.py::test_1d_manual_relu PASSED               [  6%]
tests/test_01_tensor_basics.py::test_1d_manual_sigmoid PASSED            [  7%]
tests/test_01_tensor_basics.py::test_1e_outer_sum PASSED                 [  9%]
tests/test_01_tensor_basics.py::test_no_unused_imports PASSED            [ 11%]
tests/test_02_linear_regression.py::test_2a_init_params_shapes PASSED    [ 12%]
tests/test_02_linear_regression.py::test_2b_predict_shape PASSED         [ 14%]
tests/test_02_linear_regression.py::test_2c_mse_loss PASSED              [ 15%]
tests/test_02_linear_regression.py::test_2e_train_recovers_weights PASSED [ 17%]
tests/test_02_linear_regression.py::test_no_unused_imports FAILED        [ 19%]
tests/test_03_mlp_classifier.py::test_3a_mlp_output_shape PASSED         [ 20%]
tests/test_03_mlp_classifier.py::test_3b_cross_entropy_positive PASSED   [ 22%]
tests/test_03_mlp_classifier.py::test_3c_accuracy_range PASSED           [ 23%]
tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss FAILED    [ 25%]
tests/test_03_mlp_classifier.py::test_no_unused_imports FAILED           [ 26%]
tests/test_04_cnn.py::test_4a_output_shape PASSED                        [ 28%]
tests/test_04_cnn.py::test_4b_param_count PASSED                         [ 30%]
tests/test_04_cnn.py::test_4c_forward_backward PASSED                    [ 31%]
tests/test_04_cnn.py::test_4c_rgb_input PASSED                           [ 33%]
tests/test_04_cnn.py::test_no_unused_imports FAILED                      [ 34%]
tests/test_05_attention.py::test_5a_sdpa_output_shape PASSED             [ 36%]
tests/test_05_attention.py::test_5a_weights_sum_to_one PASSED            [ 38%]
tests/test_05_attention.py::test_5b_causal_mask PASSED                   [ 39%]
tests/test_05_attention.py::test_5b_causal_output_shape PASSED           [ 41%]
tests/test_05_attention.py::test_5c_mha_output_shape PASSED              [ 42%]
tests/test_05_attention.py::test_no_unused_imports FAILED                [ 44%]
tests/test_06_training_loop.py::test_6b_cosine_lr_endpoints PASSED       [ 46%]
tests/test_06_training_loop.py::test_6b_cosine_lr_midpoint PASSED        [ 47%]
tests/test_06_training_loop.py::test_6b_cosine_lr_monotone PASSED        [ 49%]
tests/test_06_training_loop.py::test_6c_training_returns_correct_length FAILED [ 50%]
tests/test_06_training_loop.py::test_6c_training_decreases_loss FAILED   [ 52%]
tests/test_06_training_loop.py::test_no_unused_imports FAILED            [ 53%]
tests/test_07_custom_layers.py::test_7a_layernorm_shape PASSED           [ 55%]
tests/test_07_custom_layers.py::test_7a_layernorm_normalizes PASSED      [ 57%]
tests/test_07_custom_layers.py::test_7a_layernorm_learnable_params PASSED [ 58%]
tests/test_07_custom_layers.py::test_7b_embedding_shape PASSED           [ 60%]
tests/test_07_custom_layers.py::test_7b_embedding_consistent PASSED      [ 61%]
tests/test_07_custom_layers.py::test_7c_residual_block_shape PASSED      [ 63%]
tests/test_07_custom_layers.py::test_7c_residual_connection_present FAILED [ 65%]
tests/test_07_custom_layers.py::test_no_unused_imports PASSED            [ 66%]
tests/test_08_model_state.py::test_8a_state_dict_keys PASSED             [ 68%]
tests/test_08_model_state.py::test_8a_state_dict_shapes PASSED           [ 69%]
tests/test_08_model_state.py::test_8b_save_and_load_roundtrip PASSED     [ 71%]
tests/test_08_model_state.py::test_8c_copy_weights_equal_outputs PASSED  [ 73%]
tests/test_08_model_state.py::test_8d_freeze_fc1_leaves_fc2_trainable PASSED [ 74%]
tests/test_08_model_state.py::test_no_unused_imports PASSED              [ 76%]
tests/test_09_custom_losses.py::test_9a_focal_loss_shape PASSED          [ 77%]
tests/test_09_custom_losses.py::test_9a_focal_loss_positive PASSED       [ 79%]
tests/test_09_custom_losses.py::test_9a_focal_le_bce PASSED              [ 80%]
tests/test_09_custom_losses.py::test_9b_dice_loss_range PASSED           [ 82%]
tests/test_09_custom_losses.py::test_9b_dice_loss_perfect PASSED         [ 84%]
tests/test_09_custom_losses.py::test_9c_contrastive_loss_non_negative FAILED [ 85%]
tests/test_09_custom_losses.py::test_9c_contrastive_identical_similar_zero FAILED [ 87%]
tests/test_09_custom_losses.py::test_no_unused_imports PASSED            [ 88%]
tests/test_10_transformer.py::test_10a_causal_mha_shape PASSED           [ 90%]
tests/test_10_transformer.py::test_10b_ffn_shape PASSED                  [ 92%]
tests/test_10_transformer.py::test_10c_block_shape PASSED                [ 93%]
tests/test_10_transformer.py::test_10c_minilm_logits_shape PASSED        [ 95%]
tests/test_10_transformer.py::test_10d_generation_length PASSED          [ 96%]
tests/test_10_transformer.py::test_10e_training_decreases_loss PASSED    [ 98%]
tests/test_10_transformer.py::test_no_unused_imports FAILED              [100%]

=================================== FAILURES ===================================
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_02_linear_regression.py:58: AssertionError
________________________ test_3d_training_reduces_loss _________________________

    def test_3d_training_reduces_loss():
        _check_import()
        rng   = np.random.default_rng(0)
        N, D, C = 500, 16, 5
        X = rng.standard_normal((N, D)).astype(np.float32)
        Y = rng.integers(0, C, N)
        model = MLP(D, 64, C)
        optim = Adam(nn.state.get_parameters(model), lr=3e-3)
    
        first_losses, last_losses = [], []
        for step in range(300):
            idx = rng.integers(0, N, 64)
            Xb, Yb = Tensor(X[idx]), Tensor(Y[idx])
>           loss = train_step(model, optim, Xb, Yb)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_03_mlp_classifier.py:60: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_03_mlp_classifier.py:51: in train_step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f022a00bb60>

    def schedule_step(self) -> list[Tensor]:
      """
      Returns the tensors that need to be realized to perform a single optimization step.
      """
>     if not Tensor.training: raise RuntimeError(
              f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                  - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
E     RuntimeError: Tensor.training=False, Tensor.training must be enabled to use the optimizer.
E                     - help: Consider setting Tensor.training=True before calling Optimizer.step().

/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:49: RuntimeError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['Tensor', 'dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['Tensor', 'dtypes']
E       assert not ['Tensor', 'dtypes']

tests/test_03_mlp_classifier.py:74: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_04_cnn.py:57: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_05_attention.py:74: AssertionError
___________________ test_6c_training_returns_correct_length ____________________

    def test_6c_training_returns_correct_length():
        _check_import()
        rng = np.random.default_rng(3)
        N, D, C = 200, 8, 4
        X = rng.standard_normal((N, D)).astype(np.float32)
        Y = rng.integers(0, C, N)
        model = TwoLayerNet(D, 32, C)
        optim = Adam(nn.state.get_parameters(model), lr=3e-3)
>       losses = training_loop(model, optim, X, Y, epochs=10, batch_size=32)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_06_training_loop.py:46: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_06_training_loop.py:77: in training_loop
    loss = step_fn(X_batch, Y_batch)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/engine/jit.py:291: in __call__
    ret = self.fxn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
submissions/task_06_training_loop.py:26: in step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f0211955bd0>

    def schedule_step(self) -> list[Tensor]:
      """
      Returns the tensors that need to be realized to perform a single optimization step.
      """
>     if not Tensor.training: raise RuntimeError(
              f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                  - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
E     RuntimeError: Tensor.training=False, Tensor.training must be enabled to use the optimizer.
E                     - help: Consider setting Tensor.training=True before calling Optimizer.step().

/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:49: RuntimeError
_______________________ test_6c_training_decreases_loss ________________________

    def test_6c_training_decreases_loss():
        _check_import()
        rng = np.random.default_rng(3)
        N, D, C = 400, 8, 4
        X = rng.standard_normal((N, D)).astype(np.float32)
        Y = rng.integers(0, C, N)
        model = TwoLayerNet(D, 32, C)
        optim = Adam(nn.state.get_parameters(model), lr=3e-3)
>       losses = training_loop(model, optim, X, Y, epochs=50, batch_size=64)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_06_training_loop.py:58: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_06_training_loop.py:77: in training_loop
    loss = step_fn(X_batch, Y_batch)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/engine/jit.py:291: in __call__
    ret = self.fxn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
submissions/task_06_training_loop.py:26: in step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f0211956490>

    def schedule_step(self) -> list[Tensor]:
      """
      Returns the tensors that need to be realized to perform a single optimization step.
      """
>     if not Tensor.training: raise RuntimeError(
              f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                  - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
E     RuntimeError: Tensor.training=False, Tensor.training must be enabled to use the optimizer.
E                     - help: Consider setting Tensor.training=True before calling Optimizer.step().

/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:49: RuntimeError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_06_training_loop.py:63: AssertionError
_____________________ test_7c_residual_connection_present ______________________

    def test_7c_residual_connection_present():
        _check_import()
        """When fc weights are zeroed out the block should output the input unchanged."""
        d  = 8
        rb = ResidualBlock(d)
        # Zero all linear weights so FFN contributes nothing
>       for layer in [rb.fc1, rb.fc2]:
                      ^^^^^^
E       AttributeError: 'ResidualBlock' object has no attribute 'fc1'

tests/test_07_custom_layers.py:77: AttributeError
____________________ test_9c_contrastive_loss_non_negative _____________________

    def test_9c_contrastive_loss_non_negative():
        _check_import()
        N, D = 8, 16
        emb1   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
        emb2   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
        labels = Tensor(rng.integers(0, 2, N).astype(np.float32))
>       assert contrastive_loss(emb1, emb2, labels).numpy() >= 0
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_09_custom_losses.py:65: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_09_custom_losses.py:66: in contrastive_loss
    loss = (1 - labels) * 0.5 * D ** 2 + labels * 0.5 * ((margin - D).clamp(min=0) ** 2)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (8,) float> on CPU with grad None>,)
kwargs = {'min': 0}, caller = '', token = None

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
      if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
    
      if TRACEMETA >= 2:
        caller_frame = sys._getframe(frame := 1)
        caller_module = caller_frame.f_globals.get("__name__", None)
        caller_func = caller_frame.f_code.co_name
        if caller_module is None: return fn(*args, **kwargs)
    
        # if its called from nn we want to step up frames until we are out of nn
        while caller_module.startswith("tinygrad.nn") and "optim" not in caller_module:
          caller_frame = sys._getframe(frame := frame + 1)
          caller_module = caller_frame.f_globals.get("__name__", None)
          if caller_module is None: return fn(*args, **kwargs)
    
        # if its called from a lambda in tinygrad we want to look two more frames up
        if caller_module.startswith("tinygrad") and caller_func == "<lambda>": caller_frame = sys._getframe(frame := frame + 2)
        caller_module = caller_frame.f_globals.get("__name__", None)
        if caller_module is None: return fn(*args, **kwargs)
        caller_func = caller_frame.f_code.co_name
        caller_lineno = caller_frame.f_lineno
    
        caller = f"{caller_module}:{caller_lineno}::{caller_func}"
      else: caller = ""
    
      token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))
      with cpu_profile(TracingKey(fn.__name__), "USER"):
>       ret = fn(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^
E       TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:4017: TypeError
__________________ test_9c_contrastive_identical_similar_zero __________________

    def test_9c_contrastive_identical_similar_zero():
        _check_import()
        N, D = 8, 16
        same   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
        labels = Tensor(np.zeros(N, dtype=np.float32))
>       cl = contrastive_loss(same, same, labels).numpy()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_09_custom_losses.py:73: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_09_custom_losses.py:66: in contrastive_loss
    loss = (1 - labels) * 0.5 * D ** 2 + labels * 0.5 * ((margin - D).clamp(min=0) ** 2)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (8,) float> on CPU with grad None>,)
kwargs = {'min': 0}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_10_transformer.py:89: AssertionError
=========================== short test summary info ============================
FAILED tests/test_02_linear_regression.py::test_no_unused_imports - Assertion...
FAILED tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss - Runti...
FAILED tests/test_03_mlp_classifier.py::test_no_unused_imports - AssertionErr...
FAILED tests/test_04_cnn.py::test_no_unused_imports - AssertionError: Unneces...
FAILED tests/test_05_attention.py::test_no_unused_imports - AssertionError: U...
FAILED tests/test_06_training_loop.py::test_6c_training_returns_correct_length
FAILED tests/test_06_training_loop.py::test_6c_training_decreases_loss - Runt...
FAILED tests/test_06_training_loop.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_07_custom_layers.py::test_7c_residual_connection_present - ...
FAILED tests/test_09_custom_losses.py::test_9c_contrastive_loss_non_negative
FAILED tests/test_09_custom_losses.py::test_9c_contrastive_identical_similar_zero
FAILED tests/test_10_transformer.py::test_no_unused_imports - AssertionError:...
=================== 12 failed, 51 passed in 62.52s (0:01:02) ===================
