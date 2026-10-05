from copy import deepcopy
import pytest
import torch
from test_static_backbone import batches,trainer,cuda
from test_static_decoder import _model
from molvid.flow.source import FrameSourceConfig
from molvid.training.static_backbone import prepare_static_backbone_batch


def test_nonunit_source_checkpoint_contract_and_strict_resume(tmp_path):
    torch.use_deterministic_algorithms(True)
    model=_model(cuda())
    assert 'frame_source' not in model.contract()
    model.source_config=FrameSourceConfig(v_scale=.2)
    t=trainer(model);cpu,b=batches(model)
    assert t.flow.source_config==model.source_config
    assert t.contracts()['model']['frame_source']=={'h_scale':1.,'v_scale':.2}
    rng=torch.Generator(device='cuda').manual_seed(73)
    static=prepare_static_backbone_batch(model,cpu,device=cuda(),generator=t.static_corruption_generator)
    t.train_step(b,generator=rng,static_batch=static)
    assert t.successful_updates==1
    checkpoint=tmp_path/'scaled-source.pt'
    t.save_checkpoint(checkpoint,cursor={'dynamic':1,'static':1},generator=rng)
    same=trainer(deepcopy(model));same.load_checkpoint(checkpoint,generator=torch.Generator(device='cuda'))
    assert same.step==1
    wrong_model=deepcopy(model);wrong_model.source_config=FrameSourceConfig()
    wrong=trainer(wrong_model)
    with pytest.raises(ValueError):
        wrong.load_checkpoint(checkpoint,generator=torch.Generator(device='cuda'))
