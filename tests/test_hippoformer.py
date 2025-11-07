import pytest
param = pytest.mark.parametrize

import torch

def test_path_integrate():
    from hippoformer.hippoformer import PathIntegration

    path_integrator = PathIntegration(32, 64)

    actions = torch.randn(2, 16, 32)

    structure_codes = path_integrator(actions)
    structure_codes = path_integrator(actions, structure_codes) # pass in previous structure codes, it will auto use the last one as hidden and pass it to the RNN

    assert structure_codes.shape == (2, 16, 64)

@param('sensory_type', ('naive', '2d', '3d'))
@param('muon_update', (True, False))
def test_mm_tem(
    sensory_type,
    muon_update
):
    import torch
    from hippoformer.hippoformer import mmTEM

    from torch.nn import Linear

    if sensory_type == 'naive':
        enc_dec = (
            Linear(11, 32),
            Linear(32, 11)
        )
        sensory = torch.randn(2, 16, 11)

    elif sensory_type == '2d':

        from hippoformer.hippoformer import grid_sensory_enc_dec

        enc_dec = grid_sensory_enc_dec
        sensory = torch.randn(2, 16, 9)

    elif sensory_type == '3d':

        from hippoformer.hippoformer import maze_sensory_enc_dec

        enc_dec = maze_sensory_enc_dec

        sensory = torch.randn(2, 3, 16, 64, 64)

    model = mmTEM(
        dim = 32,
        sensory_encoder_decoder = enc_dec,
        dim_sensory = 11,
        dim_action = 7,
        dim_structure = 32,
        dim_encoded_sensory = 32,
        muon_update = muon_update
    )

    actions = torch.randn(2, 16, 7)

    next_params = model(sensory, actions, return_memory_mlp_params = True)
    next_params = model(sensory, actions, memory_mlp_params = next_params, return_memory_mlp_params = True)

    loss = model(sensory, actions, memory_mlp_params = next_params)
    loss.backward()

def test_tem_t():
    from hippoformer.hippoformer import TEMTransformerBlock

    block = TEMTransformerBlock(32, 16, window_size = 3)

    structural_codes = torch.randn(1, 7, 32)
    encoded_sensory = torch.randn(1, 7, 16)

    pred, kv_cache = block(structural_codes, encoded_sensory)

    assert pred.shape == (1, 7, 32)