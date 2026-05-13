## Neural Network classes

#### nn.BatchNorm(sz: 'int', eps=1e-05, affine=True, track_running_stats=True, momentum=0.1)


#### nn.Conv1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding: 'int | str' = 0, dilation=1, groups=1, bias=True) -> Conv2d


#### nn.Conv2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding: 'int | tuple[int, ...] | str' = 0, dilation=1, groups=1, bias=True)


#### nn.ConvTranspose1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True) -> ConvTranspose2d


#### nn.ConvTranspose2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True)


#### nn.Linear(in_features: 'int', out_features: 'int', bias=True)


#### nn.GroupNorm(num_groups: 'int', num_channels: 'int', eps=1e-05, affine=True)


#### nn.InstanceNorm(num_features: 'int', eps: 'float' = 1e-05, affine: 'bool' = True)


#### nn.LayerNorm(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)


#### nn.LayerNorm2d(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)


#### nn.RMSNorm(dim: 'int', eps=1e-06, elementwise_affine=True)


#### nn.Embedding(vocab_size: 'int', embed_size: 'int')


#### nn.LSTMCell(input_size: 'int', hidden_size: 'int', bias: 'bool' = True)


## Optimizers

#### nn.optim.SGD(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, fused=<ContextVar>)


#### nn.optim.LARS(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.9, weight_decay=0.0001, ns_steps=0, ns_coefficients=None, nesterov=False, classic=True, pre_wd=True, tcoef=0.001, fused=<ContextVar>)


#### nn.optim.AdamW(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, weight_decay=0.01, fused=<ContextVar>)


#### nn.optim.Adam(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, fused=<ContextVar>)


#### nn.optim.LAMB(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-06, weight_decay=0.0, adam=False, fused=<ContextVar>)


## Load/Save

#### nn.state.safe_load(fn: tinygrad.tensor.Tensor | str | pathlib.Path) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.safe_save(tensors: dict[str, tinygrad.tensor.Tensor], fn: str, metadata: dict[str, Any] | None = None)


#### nn.state.get_state_dict(obj, prefix: str = '', tensor_type=<class 'tinygrad.tensor.Tensor'>) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.get_parameters(obj) -> list[tinygrad.tensor.Tensor]


#### nn.state.load_state_dict(model, state_dict: dict[str, tinygrad.tensor.Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[tinygrad.tensor.Tensor]


#### nn.state.tar_extract(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.torch_load(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.gguf_load(tensor: tinygrad.tensor.Tensor) -> tuple[dict, dict[str, tinygrad.tensor.Tensor]]

