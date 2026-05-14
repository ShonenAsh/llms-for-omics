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
tests/test_02_linear_regression.py::test_2e_train_recovers_weights FAILED [ 17%]
tests/test_02_linear_regression.py::test_no_unused_imports PASSED        [ 19%]
tests/test_03_mlp_classifier.py::test_3a_mlp_output_shape PASSED         [ 20%]
tests/test_03_mlp_classifier.py::test_3b_cross_entropy_positive PASSED   [ 22%]
tests/test_03_mlp_classifier.py::test_3c_accuracy_range PASSED           [ 23%]
tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss PASSED    [ 25%]
tests/test_03_mlp_classifier.py::test_no_unused_imports FAILED           [ 26%]
tests/test_04_cnn.py::test_4a_output_shape PASSED                        [ 28%]
tests/test_04_cnn.py::test_4b_param_count PASSED                         [ 30%]
tests/test_04_cnn.py::test_4c_forward_backward PASSED                    [ 31%]
tests/test_04_cnn.py::test_4c_rgb_input PASSED                           [ 33%]
tests/test_04_cnn.py::test_no_unused_imports FAILED                      [ 34%]
tests/test_05_attention.py::test_5a_sdpa_output_shape PASSED             [ 36%]
tests/test_05_attention.py::test_5a_weights_sum_to_one PASSED            [ 38%]
tests/test_05_attention.py::test_5b_causal_mask FAILED                   [ 39%]
tests/test_05_attention.py::test_5b_causal_output_shape FAILED           [ 41%]
tests/test_05_attention.py::test_5c_mha_output_shape PASSED              [ 42%]
tests/test_05_attention.py::test_no_unused_imports PASSED                [ 44%]
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
tests/test_10_transformer.py::test_10d_generation_length FAILED          [ 96%]
tests/test_10_transformer.py::test_10e_training_decreases_loss PASSED    [ 98%]
tests/test_10_transformer.py::test_no_unused_imports PASSED              [100%]

=================================== FAILURES ===================================
________________________ test_2e_train_recovers_weights ________________________

    def test_2e_train_recovers_weights():
        _check_import()
        rng = np.random.default_rng(42)
        N, D = 200, 3
        true_w = np.array([1.5, -2.0, 0.5], dtype=np.float32)
        true_b = np.float32(0.3)
        X = rng.standard_normal((N, D)).astype(np.float32)
        Y = X @ true_w + true_b + rng.standard_normal(N).astype(np.float32) * 0.05
    
>       w, b = train(X, Y, lr=0.05, steps=500)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_02_linear_regression.py:52: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_02_linear_regression.py:55: in train
    w, b = sgd_step(w, b, lr)
           ^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

w = <Tensor <UOp CPU (3,) float> on CPU with grad None>
b = <Tensor <UOp CPU () float> on CPU with grad None>, lr = 0.05

    def sgd_step(w: Tensor, b: Tensor, lr: float) -> tuple[Tensor, Tensor]:
        """
        Update w and b in-place using their gradients and learning rate lr.
        Use .assign() to keep the computation graph clean. Return (w, b).
        """
>       w.assign(w - lr * w.grad)
                     ^^^^^^^^^^^
E       TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'

submissions/task_02_linear_regression.py:32: TypeError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['Adam', 'dtypes', 'np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['Adam', 'dtypes', 'np']
E       assert not ['Adam', 'dtypes', 'np']

tests/test_03_mlp_classifier.py:74: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_04_cnn.py:57: AssertionError
_____________________________ test_5b_causal_mask ______________________________

    def test_5b_causal_mask():
        _check_import()
        B, T, d = 2, 6, 8
        Q = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        K = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        V = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
>       _, w = causal_attention(Q, K, V)
               ^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_05_attention.py:46: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Q = <Tensor <UOp CPU (2, 6, 8) float> on CPU with grad None>
K = <Tensor <UOp CPU (2, 6, 8) float> on CPU with grad None>
V = <Tensor <UOp CPU (2, 6, 8) float> on CPU with grad None>

    def causal_attention(Q: Tensor, K: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
        """
        Self-attention where position i may only attend to positions j <= i.
        Q, K, V: (B, T, d_k)
        Returns: (context, attn_weights) both shape (B, T, T).
        """
        B, T, _ = Q.shape
        d_k = Q.shape[-1]
        scores = Q.matmul(K.transpose(1, 2)) * (d_k ** -0.5)
    
        # Create causal mask: positions j > i get -inf
        # Lower triangular (including diagonal) stays as is, upper off-diagonal becomes -inf
>       mask = Tensor.ones((1, T, T), dtype=Tensor.default_dtype).tril(0)
                                            ^^^^^^^^^^^^^^^^^^^^
E       AttributeError: type object 'Tensor' has no attribute 'default_dtype'

submissions/task_05_attention.py:29: AttributeError
_________________________ test_5b_causal_output_shape __________________________

    def test_5b_causal_output_shape():
        _check_import()
        B, T, d = 2, 5, 8
        Q = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        K = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
        V = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
>       ctx, _ = causal_attention(Q, K, V)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_05_attention.py:60: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Q = <Tensor <UOp CPU (2, 5, 8) float> on CPU with grad None>
K = <Tensor <UOp CPU (2, 5, 8) float> on CPU with grad None>
V = <Tensor <UOp CPU (2, 5, 8) float> on CPU with grad None>

    def causal_attention(Q: Tensor, K: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
        """
        Self-attention where position i may only attend to positions j <= i.
        Q, K, V: (B, T, d_k)
        Returns: (context, attn_weights) both shape (B, T, T).
        """
        B, T, _ = Q.shape
        d_k = Q.shape[-1]
        scores = Q.matmul(K.transpose(1, 2)) * (d_k ** -0.5)
    
        # Create causal mask: positions j > i get -inf
        # Lower triangular (including diagonal) stays as is, upper off-diagonal becomes -inf
>       mask = Tensor.ones((1, T, T), dtype=Tensor.default_dtype).tril(0)
                                            ^^^^^^^^^^^^^^^^^^^^
E       AttributeError: type object 'Tensor' has no attribute 'default_dtype'

submissions/task_05_attention.py:29: AttributeError
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
submissions/task_06_training_loop.py:78: in training_loop
    loss_tensor = step(X_batch, Y_batch)
                  ^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/engine/jit.py:291: in __call__
    ret = self.fxn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
submissions/task_06_training_loop.py:26: in step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f55acc3c410>

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
submissions/task_06_training_loop.py:78: in training_loop
    loss_tensor = step(X_batch, Y_batch)
                  ^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/engine/jit.py:291: in __call__
    ret = self.fxn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
submissions/task_06_training_loop.py:26: in step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f55acc3e5d0>

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
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

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
submissions/task_09_custom_losses.py:22: in focal_loss
    p_t = p_t.clamp(min=eps, max=1 - eps)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (50,) float> on CPU with grad None>,)
kwargs = {'max': 0.9999999, 'min': 1e-07}, caller = '', token = None

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
submissions/task_09_custom_losses.py:22: in focal_loss
    p_t = p_t.clamp(min=eps, max=1 - eps)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (100,) float> on CPU with grad None>,)
kwargs = {'max': 0.9999999, 'min': 1e-07}

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
submissions/task_09_custom_losses.py:22: in focal_loss
    p_t = p_t.clamp(min=eps, max=1 - eps)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (100,) float> on CPU with grad None>,)
kwargs = {'max': 0.9999999, 'min': 1e-07}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
__________________________ test_10d_generation_length __________________________

    def test_10d_generation_length():
        _check_import()
        model   = make_model()
        Tensor.training = False
        prompt  = Tensor(rng.integers(0, VOCAB, (1, 4)))
>       out     = model.generate(prompt, max_new_tokens=6)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_10_transformer.py:61: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_10_transformer.py:112: in generate
    idx = Tensor.cat([idx, next_token], dim=1)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: in _wrapper
    if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                            ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = [<Tensor <UOp CPU (1, 4) long> on CPU with grad None>, <Tensor <UOp CPU (1, 1) int> on CPU with grad None>]
dim = 1, args = ()

    def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
      """
      Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
      All tensors must have the same shape except in the concatenating dimension.
    
      ```python exec="true" source="above" session="tensor" result="python"
      t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
      print(t0.cat(t1, t2, dim=0).numpy())
      ```
      ```python exec="true" source="above" session="tensor" result="python"
      print(t0.cat(t1, t2, dim=1).numpy())
      ```
      """
>     dim = self._resolve_dim(dim)
            ^^^^^^^^^^^^^^^^^
E     AttributeError: 'list' object has no attribute '_resolve_dim'

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:1308: AttributeError
=========================== short test summary info ============================
FAILED tests/test_02_linear_regression.py::test_2e_train_recovers_weights - T...
FAILED tests/test_03_mlp_classifier.py::test_no_unused_imports - AssertionErr...
FAILED tests/test_04_cnn.py::test_no_unused_imports - AssertionError: Unneces...
FAILED tests/test_05_attention.py::test_5b_causal_mask - AttributeError: type...
FAILED tests/test_05_attention.py::test_5b_causal_output_shape - AttributeErr...
FAILED tests/test_06_training_loop.py::test_6c_training_returns_correct_length
FAILED tests/test_06_training_loop.py::test_6c_training_decreases_loss - Runt...
FAILED tests/test_06_training_loop.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_07_custom_layers.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_09_custom_losses.py::test_9a_focal_loss_shape - TypeError: ...
FAILED tests/test_09_custom_losses.py::test_9a_focal_loss_positive - TypeErro...
FAILED tests/test_09_custom_losses.py::test_9a_focal_le_bce - TypeError: Math...
FAILED tests/test_10_transformer.py::test_10d_generation_length - AttributeEr...
=================== 13 failed, 50 passed in 85.47s (0:01:25) ===================
