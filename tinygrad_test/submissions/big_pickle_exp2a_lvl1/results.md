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
tests/test_05_attention.py::test_5b_causal_mask FAILED                   [ 39%]
tests/test_05_attention.py::test_5b_causal_output_shape PASSED           [ 41%]
tests/test_05_attention.py::test_5c_mha_output_shape PASSED              [ 42%]
tests/test_05_attention.py::test_no_unused_imports PASSED                [ 44%]
tests/test_06_training_loop.py::test_6b_cosine_lr_endpoints FAILED       [ 46%]
tests/test_06_training_loop.py::test_6b_cosine_lr_midpoint FAILED        [ 47%]
tests/test_06_training_loop.py::test_6b_cosine_lr_monotone FAILED        [ 49%]
tests/test_06_training_loop.py::test_6c_training_returns_correct_length FAILED [ 50%]
tests/test_06_training_loop.py::test_6c_training_decreases_loss FAILED   [ 52%]
tests/test_06_training_loop.py::test_no_unused_imports PASSED            [ 53%]
tests/test_07_custom_layers.py::test_7a_layernorm_shape PASSED           [ 55%]
tests/test_07_custom_layers.py::test_7a_layernorm_normalizes PASSED      [ 57%]
tests/test_07_custom_layers.py::test_7a_layernorm_learnable_params PASSED [ 58%]
tests/test_07_custom_layers.py::test_7b_embedding_shape PASSED           [ 60%]
tests/test_07_custom_layers.py::test_7b_embedding_consistent PASSED      [ 61%]
tests/test_07_custom_layers.py::test_7c_residual_block_shape PASSED      [ 63%]
tests/test_07_custom_layers.py::test_7c_residual_connection_present PASSED [ 65%]
tests/test_07_custom_layers.py::test_no_unused_imports FAILED            [ 66%]
tests/test_08_model_state.py::test_8a_state_dict_keys PASSED             [ 68%]
tests/test_08_model_state.py::test_8a_state_dict_shapes PASSED           [ 69%]
tests/test_08_model_state.py::test_8b_save_and_load_roundtrip PASSED     [ 71%]
tests/test_08_model_state.py::test_8c_copy_weights_equal_outputs PASSED  [ 73%]
tests/test_08_model_state.py::test_8d_freeze_fc1_leaves_fc2_trainable PASSED [ 74%]
tests/test_08_model_state.py::test_no_unused_imports PASSED              [ 76%]
tests/test_09_custom_losses.py::test_9a_focal_loss_shape FAILED          [ 77%]
tests/test_09_custom_losses.py::test_9a_focal_loss_positive FAILED       [ 79%]
tests/test_09_custom_losses.py::test_9a_focal_le_bce FAILED              [ 80%]
tests/test_09_custom_losses.py::test_9b_dice_loss_range PASSED           [ 82%]
tests/test_09_custom_losses.py::test_9b_dice_loss_perfect PASSED         [ 84%]
tests/test_09_custom_losses.py::test_9c_contrastive_loss_non_negative PASSED [ 85%]
tests/test_09_custom_losses.py::test_9c_contrastive_identical_similar_zero PASSED [ 87%]
tests/test_09_custom_losses.py::test_no_unused_imports PASSED            [ 88%]
tests/test_10_transformer.py::test_10a_causal_mha_shape PASSED           [ 90%]
tests/test_10_transformer.py::test_10b_ffn_shape PASSED                  [ 92%]
tests/test_10_transformer.py::test_10c_block_shape PASSED                [ 93%]
tests/test_10_transformer.py::test_10c_minilm_logits_shape PASSED        [ 95%]
tests/test_10_transformer.py::test_10d_generation_length PASSED          [ 96%]
tests/test_10_transformer.py::test_10e_training_decreases_loss PASSED    [ 98%]
tests/test_10_transformer.py::test_no_unused_imports PASSED              [100%]

=================================== FAILURES ===================================
____________________________ test_no_unused_imports ____________________________

unused_imports = ['np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['np']
E       assert not ['np']

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
submissions/task_03_mlp_classifier.py:41: in train_step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f8ff912be00>

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

unused_imports = ['np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['np']
E       assert not ['np']

tests/test_03_mlp_classifier.py:74: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['np']
E       assert not ['np']

tests/test_04_cnn.py:57: AssertionError
_____________________________ test_5b_causal_mask ______________________________

    def test_5b_causal_mask():
        _check_import()
        B, T, d = 2, 6, 8
        Q = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        K = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        V = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        _, w = causal_attention(Q, K, V)
        w_np = w.numpy()
        for b in range(B):
            for i in range(T):
                for j in range(i + 1, T):
>                   assert w_np[b, i, j] < 1e-5, f"future ({i},{j}) not masked: {w_np[b,i,j]}"
E                   AssertionError: future (0,1) not masked: 0.11917883902788162
E                   assert np.float32(0.11917884) < 1e-05

tests/test_05_attention.py:51: AssertionError
_________________________ test_6b_cosine_lr_endpoints __________________________

    def test_6b_cosine_lr_endpoints():
>       _check_import()

tests/test_06_training_loop.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ModuleNotFoundError: No module named 'tinygrad.jit'

tests/test_06_training_loop.py:14: Failed
__________________________ test_6b_cosine_lr_midpoint __________________________

    def test_6b_cosine_lr_midpoint():
>       _check_import()

tests/test_06_training_loop.py:26: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ModuleNotFoundError: No module named 'tinygrad.jit'

tests/test_06_training_loop.py:14: Failed
__________________________ test_6b_cosine_lr_monotone __________________________

    def test_6b_cosine_lr_monotone():
>       _check_import()

tests/test_06_training_loop.py:33: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ModuleNotFoundError: No module named 'tinygrad.jit'

tests/test_06_training_loop.py:14: Failed
___________________ test_6c_training_returns_correct_length ____________________

    def test_6c_training_returns_correct_length():
>       _check_import()

tests/test_06_training_loop.py:39: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ModuleNotFoundError: No module named 'tinygrad.jit'

tests/test_06_training_loop.py:14: Failed
_______________________ test_6c_training_decreases_loss ________________________

    def test_6c_training_decreases_loss():
>       _check_import()

tests/test_06_training_loop.py:51: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ModuleNotFoundError: No module named 'tinygrad.jit'

tests/test_06_training_loop.py:14: Failed
____________________________ test_no_unused_imports ____________________________

unused_imports = ['np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['np']
E       assert not ['np']

tests/test_07_custom_layers.py:87: AssertionError
___________________________ test_9a_focal_loss_shape ___________________________

    def test_9a_focal_loss_shape():
        _check_import()
        logits  = Tensor(rng.standard_normal(50).astype(np.float32))
        targets = Tensor(rng.integers(0, 2, 50).astype(np.float32))
>       loss = focal_loss(logits, targets)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_09_custom_losses.py:23: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_09_custom_losses.py:24: in focal_loss
    p_t = p_t.clamp(min=1e-12)
          ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (50,) float> on CPU with grad None>,)
kwargs = {'min': 1e-12}, caller = '', token = None

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
_________________________ test_9a_focal_loss_positive __________________________

    def test_9a_focal_loss_positive():
        _check_import()
        logits  = Tensor(rng.standard_normal(100).astype(np.float32))
        targets = Tensor(rng.integers(0, 2, 100).astype(np.float32))
>       assert focal_loss(logits, targets).numpy() > 0
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_09_custom_losses.py:31: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_09_custom_losses.py:24: in focal_loss
    p_t = p_t.clamp(min=1e-12)
          ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (100,) float> on CPU with grad None>,)
kwargs = {'min': 1e-12}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
_____________________________ test_9a_focal_le_bce _____________________________

    def test_9a_focal_le_bce():
        _check_import()
        logits  = Tensor(rng.standard_normal(100).astype(np.float32))
        targets = Tensor(rng.integers(0, 2, 100).astype(np.float32))
>       fl  = focal_loss(logits, targets).numpy()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_09_custom_losses.py:38: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_09_custom_losses.py:24: in focal_loss
    p_t = p_t.clamp(min=1e-12)
          ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (100,) float> on CPU with grad None>,)
kwargs = {'min': 1e-12}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
=========================== short test summary info ============================
FAILED tests/test_02_linear_regression.py::test_no_unused_imports - Assertion...
FAILED tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss - Runti...
FAILED tests/test_03_mlp_classifier.py::test_no_unused_imports - AssertionErr...
FAILED tests/test_04_cnn.py::test_no_unused_imports - AssertionError: Unneces...
FAILED tests/test_05_attention.py::test_5b_causal_mask - AssertionError: futu...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_endpoints - Failed: ...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_midpoint - Failed: C...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_monotone - Failed: C...
FAILED tests/test_06_training_loop.py::test_6c_training_returns_correct_length
FAILED tests/test_06_training_loop.py::test_6c_training_decreases_loss - Fail...
FAILED tests/test_07_custom_layers.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_09_custom_losses.py::test_9a_focal_loss_shape - TypeError: ...
FAILED tests/test_09_custom_losses.py::test_9a_focal_loss_positive - TypeErro...
FAILED tests/test_09_custom_losses.py::test_9a_focal_le_bce - TypeError: Math...
=================== 14 failed, 49 passed in 65.43s (0:01:05) ===================
