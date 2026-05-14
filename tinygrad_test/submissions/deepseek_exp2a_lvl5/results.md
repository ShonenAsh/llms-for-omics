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
tests/test_02_linear_regression.py::test_no_unused_imports PASSED        [ 19%]
tests/test_03_mlp_classifier.py::test_3a_mlp_output_shape PASSED         [ 20%]
tests/test_03_mlp_classifier.py::test_3b_cross_entropy_positive PASSED   [ 22%]
tests/test_03_mlp_classifier.py::test_3c_accuracy_range PASSED           [ 23%]
tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss FAILED    [ 25%]
tests/test_03_mlp_classifier.py::test_no_unused_imports PASSED           [ 26%]
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
tests/test_07_custom_layers.py::test_7c_residual_connection_present FAILED [ 65%]
tests/test_07_custom_layers.py::test_no_unused_imports PASSED            [ 66%]
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
tests/test_09_custom_losses.py::test_9c_contrastive_loss_non_negative PASSED [ 85%]
tests/test_09_custom_losses.py::test_9c_contrastive_identical_similar_zero PASSED [ 87%]
tests/test_09_custom_losses.py::test_no_unused_imports FAILED            [ 88%]
tests/test_10_transformer.py::test_10a_causal_mha_shape PASSED           [ 90%]
tests/test_10_transformer.py::test_10b_ffn_shape PASSED                  [ 92%]
tests/test_10_transformer.py::test_10c_block_shape PASSED                [ 93%]
tests/test_10_transformer.py::test_10c_minilm_logits_shape PASSED        [ 95%]
tests/test_10_transformer.py::test_10d_generation_length PASSED          [ 96%]
tests/test_10_transformer.py::test_10e_training_decreases_loss FAILED    [ 98%]
tests/test_10_transformer.py::test_no_unused_imports PASSED              [100%]

=================================== FAILURES ===================================
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
submissions/task_03_mlp_classifier.py:52: in train_step
    optim.step()
/usr/local/lib/python3.13/site-packages/tinygrad/nn/optim.py:43: in step
    Tensor.realize(*self.schedule_step())
                    ^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tinygrad.nn.optim.LAMB object at 0x7f44645963c0>

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
E                   AssertionError: future (0,1) not masked: nan
E                   assert np.float32(nan) < 1e-05

tests/test_05_attention.py:51: AssertionError
_________________________ test_6b_cosine_lr_endpoints __________________________

    def test_6b_cosine_lr_endpoints():
>       _check_import()

tests/test_06_training_loop.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ImportError: cannot import name 'TwoLayerNet' from 'task_06_training_loop' (/workspace/submissions/task_06_training_loop.py)

tests/test_06_training_loop.py:14: Failed
__________________________ test_6b_cosine_lr_midpoint __________________________

    def test_6b_cosine_lr_midpoint():
>       _check_import()

tests/test_06_training_loop.py:26: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ImportError: cannot import name 'TwoLayerNet' from 'task_06_training_loop' (/workspace/submissions/task_06_training_loop.py)

tests/test_06_training_loop.py:14: Failed
__________________________ test_6b_cosine_lr_monotone __________________________

    def test_6b_cosine_lr_monotone():
>       _check_import()

tests/test_06_training_loop.py:33: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ImportError: cannot import name 'TwoLayerNet' from 'task_06_training_loop' (/workspace/submissions/task_06_training_loop.py)

tests/test_06_training_loop.py:14: Failed
___________________ test_6c_training_returns_correct_length ____________________

    def test_6c_training_returns_correct_length():
>       _check_import()

tests/test_06_training_loop.py:39: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ImportError: cannot import name 'TwoLayerNet' from 'task_06_training_loop' (/workspace/submissions/task_06_training_loop.py)

tests/test_06_training_loop.py:14: Failed
_______________________ test_6c_training_decreases_loss ________________________

    def test_6c_training_decreases_loss():
>       _check_import()

tests/test_06_training_loop.py:51: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _check_import():
        if _import_error is not None:
>           pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")
E           Failed: Could not import task module: ImportError: cannot import name 'TwoLayerNet' from 'task_06_training_loop' (/workspace/submissions/task_06_training_loop.py)

tests/test_06_training_loop.py:14: Failed
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
____________________________ test_no_unused_imports ____________________________

unused_imports = ['Tensor']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['Tensor']
E       assert not ['Tensor']

tests/test_08_model_state.py:72: AssertionError
____________________________ test_no_unused_imports ____________________________

unused_imports = ['dtypes', 'math']

    def test_no_unused_imports(unused_imports):
>       assert not unused_imports, f"Unnecessary imports: {unused_imports}"
E       AssertionError: Unnecessary imports: ['dtypes', 'math']
E       assert not ['dtypes', 'math']

tests/test_09_custom_losses.py:78: AssertionError
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
E       AssertionError: loss didn't decrease: nan → nan
E       assert nan < nan

tests/test_10_transformer.py:85: AssertionError
=========================== short test summary info ============================
FAILED tests/test_03_mlp_classifier.py::test_3d_training_reduces_loss - Runti...
FAILED tests/test_04_cnn.py::test_no_unused_imports - AssertionError: Unneces...
FAILED tests/test_05_attention.py::test_5b_causal_mask - AssertionError: futu...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_endpoints - Failed: ...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_midpoint - Failed: C...
FAILED tests/test_06_training_loop.py::test_6b_cosine_lr_monotone - Failed: C...
FAILED tests/test_06_training_loop.py::test_6c_training_returns_correct_length
FAILED tests/test_06_training_loop.py::test_6c_training_decreases_loss - Fail...
FAILED tests/test_07_custom_layers.py::test_7c_residual_connection_present - ...
FAILED tests/test_08_model_state.py::test_no_unused_imports - AssertionError:...
FAILED tests/test_09_custom_losses.py::test_no_unused_imports - AssertionErro...
FAILED tests/test_10_transformer.py::test_10e_training_decreases_loss - Asser...
=================== 12 failed, 51 passed in 65.07s (0:01:05) ===================
