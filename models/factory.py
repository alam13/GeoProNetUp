from model import (
    Net_coor,
    Net_coor_cent,
    Net_coor_dir,
    Net_coor_len,
    Net_coor_res,
    Net_coor_torsion,
    Net_coor_two_stage,
)


MODEL_REGISTRY = {
    "Net_coor": Net_coor,
    "Net_coor_res": Net_coor_res,
    "Net_coor_dir": Net_coor_dir,
    "Net_coor_len": Net_coor_len,
    "Net_coor_cent": Net_coor_cent,
    "Net_coor_torsion": Net_coor_torsion,
    "Net_coor_two_stage": Net_coor_two_stage,
}


def build_model(model_type: str, in_channels: int, args, device):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_type](in_channels, args)
    return model.to(device)
