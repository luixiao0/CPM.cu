python3 setup.py install
cd tests
python3 test.py

|weight|
|weight|embed |
|weight|hidden|
|weight|hidden|norm  |
|weight|hidden|norm  |q     |k     |v     |
|weight|hidden|norm  |q     |k     |v     |attn_output, softmax_lse, etc.
|weight|hidden| # hidden = attn_output @ o_proj + hidden
|weight|hidden|norm  |
|weight|hidden|norm  |gate  |up    |
|weight|hidden|norm  |gate  |up    |silu_gated_up
|weight|hidden| # hidden = silu_gated_up @ down_proj + hidden

