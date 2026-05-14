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
tests/test_05_attention.py::test_no_unused_imports PASSED                [ 44%]
tests/test_06_training_loop.py::test_6b_cosine_lr_endpoints PASSED       [ 46%]
tests/test_06_training_loop.py::test_6b_cosine_lr_midpoint PASSED        [ 47%]
tests/test_06_training_loop.py::test_6b_cosine_lr_monotone PASSED        [ 49%]
tests/test_06_training_loop.py::test_6c_training_returns_correct_length PASSED [ 50%]
tests/test_06_training_loop.py::test_6c_training_decreases_loss PASSED   [ 52%]
tests/test_06_training_loop.py::test_no_unused_imports PASSED            [ 53%]
tests/test_07_custom_layers.py::test_7a_layernorm_shape PASSED           [ 55%]
tests/test_07_custom_layers.py::test_7a_layernorm_normalizes PASSED      [ 57%]
tests/test_07_custom_layers.py::test_7a_layernorm_learnable_params PASSED [ 58%]
tests/test_07_custom_layers.py::test_7b_embedding_shape PASSED           [ 60%]
tests/test_07_custom_layers.py::test_7b_embedding_consistent PASSED      [ 61%]
tests/test_07_custom_layers.py::test_7c_residual_block_shape FAILED      [ 63%]
tests/test_07_custom_layers.py::test_7c_residual_connection_present FAILED [ 65%]
tests/test_07_custom_layers.py::test_no_unused_imports FAILED            [ 66%]
tests/test_08_model_state.py::test_8a_state_dict_keys PASSED             [ 68%]
tests/test_08_model_state.py::test_8a_state_dict_shapes PASSED           [ 69%]
tests/test_08_model_state.py::test_8b_save_and_load_roundtrip PASSED     [ 71%]
tests/test_08_model_state.py::test_8c_copy_weights_equal_outputs PASSED  [ 73%]
tests/test_08_model_state.py::test_8d_freeze_fc1_leaves_fc2_trainable PASSED [ 74%]
tests/test_08_model_state.py::test_no_unused_imports FAILED              [ 76%]
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
tests/test_10_transformer.py::test_10e_training_decreases_loss FAILED    [ 98%]
tests/test_10_transformer.py::test_no_unused_imports FAILED              [100%]

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
submissions/task_03_mlp_classifier.py:37: in train_step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f52d9c43620>

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

unused_imports = ['Tensor', 'dtypes', 'np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['Tensor', 'dtypes', 'np']
E       assert not ['Tensor', 'dtypes', 'np']

tests/test_03_mlp_classifier.py:74: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes', 'np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes', 'np']
E       assert not ['dtypes', 'np']

tests/test_04_cnn.py:57: AssertionError
_________________________ test_7c_residual_block_shape _________________________

    def test_7c_residual_block_shape():
        _check_import()
        d = 64
        rb = ResidualBlock(d)
        x  = Tensor(rng.standard_normal((2, 6, d)).astype(np.float32))
>       assert rb(x).shape == (2, 6, d)
               ^^^^^

tests/test_07_custom_layers.py:68: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
submissions/task_07_custom_layers.py:43: in __call__
    x = x.linear(self.fc1.transpose(), self.bias1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:4017: in _wrapper
    ret = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3373: in linear
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
                                                        ^^^^^^^^^^^^^^^^
/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: in _wrapper
    if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                            ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <Tensor <UOp CPU (2, 6, 64) float> on CPU with grad None>
w = <Tensor <UOp CPU (256, 64) float> on CPU with grad None>, dtype = None

    def dot(self, w:Tensor, dtype:DTypeLike|None=None) -> Tensor:
    
      """
      Performs dot product between two tensors.
      If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
      If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.
    
      You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.
    
      ```python exec="true" source="above" session="tensor" result="python"
      a = Tensor([1, 2, 3])
      b = Tensor([1, 1, 0])
      print(a.dot(b).numpy())
      ```
      ```python exec="true" source="above" session="tensor" result="python"
      a = Tensor([[1, 2], [3, 4]])
      b = Tensor([[5, 6], [7, 8]])
      print(a.dot(b).numpy())
      ```
      """
      if IMAGE: return self.image_dot(w, dtype)
      x, dx, dw = self, self.ndim, w.ndim
      if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
>     if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     RuntimeError: cannot dot (2, 6, 64) and (256, 64)

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:2378: RuntimeError
_____________________ test_7c_residual_connection_present ______________________

    def test_7c_residual_connection_present():
        _check_import()
        """When fc weights are zeroed out the block should output the input unchanged."""
        d  = 8
        rb = ResidualBlock(d)
        # Zero all linear weights so FFN contributes nothing
        for layer in [rb.fc1, rb.fc2]:
>           layer.weight.assign(Tensor.zeros_like(layer.weight))
            ^^^^^^^^^^^^
E           AttributeError: 'Tensor' object has no attribute 'weight'

tests/test_07_custom_layers.py:78: AttributeError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes']
E       assert not ['dtypes']

tests/test_07_custom_layers.py:87: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['get_parameters', 'np']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['get_parameters', 'np']
E       assert not ['get_parameters', 'np']

tests/test_08_model_state.py:72: AssertionError
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
submissions/task_09_custom_losses.py:65: in contrastive_loss
    dissimilar_term = labels * 0.5 * (margin - D).clamp(min=0).square()
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (8,) float> on CPU with grad None>,)
kwargs = {'min': 0}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
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
submissions/task_09_custom_losses.py:65: in contrastive_loss
    dissimilar_term = labels * 0.5 * (margin - D).clamp(min=0).square()
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<Tensor <UOp CPU (8,) float> on CPU with grad None>,)
kwargs = {'min': 0}

    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
>     if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^
E     TypeError: MathMixin.clamp() got an unexpected keyword argument 'min'. Did you mean 'min_'?

/usr/local/lib/python3.13/site-packages/tinygrad/tensor.py:3991: TypeError
_______________________ test_10e_training_decreases_loss _______________________

    def test_10e_training_decreases_loss():
        _check_import()
        model = make_model()
        optim = Adam(nn.state.get_parameters(model), lr=3e-3)
    
        def step(x, y):
            Tensor.training = True
            optim.zero_grad()
            loss = model(x).reshape(-1, VOCAB).sparse_categorical_crossentropy(y.reshape(-1))
            loss.backward()
            optim.step()
            return loss
    
        jit_step = TinyJit(step)
        losses = []
        for _ in range(60):
            seq = Tensor(rng.integers(0, VOCAB, (B, SEQ)))
            tgt = Tensor(rng.integers(0, VOCAB, (B, SEQ)))
            losses.append(jit_step(seq, tgt).numpy().item())
    
>       assert losses[-1] < losses[0], f"loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
E       AssertionError: loss didn't decrease: 3.5049 → 3.5348
E       assert 3.5347630977630615 < 3.504906415939331

tests/test_10_transformer.py:85: AssertionError
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
FAILED tests/test_07_custom_layers.py::test_7c_residual_block_shape - Runtime...
FAILED tests/test_07_custom_layers.py::test_7c_residual_connection_present - ...
FAILED tests/test_07_custom_layers.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_08_model_state.py::test_no_unused_imports - AssertionError:...
FAILED tests/test_09_custom_losses.py::test_9c_contrastive_loss_non_negative
FAILED tests/test_09_custom_losses.py::test_9c_contrastive_identical_similar_zero
FAILED tests/test_10_transformer.py::test_10e_training_decreases_loss - Asser...
FAILED tests/test_10_transformer.py::test_no_unused_imports - AssertionError:...
=================== 12 failed, 51 passed in 77.01s (0:01:17) ===================
