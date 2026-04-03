from dataclasses import dataclass
from typing import Optional

import torch

from model import Net_coor, Net_coor_torsion, Net_coor_two_stage


@dataclass
class InferenceConfig:
    model_type: str = "Net_coor"
    edge_dim: int = 13
    use_alpha_channel: bool = False


class PoseInferencePipeline:
    """
    Deterministic inference wrapper for preprocessing + forward + postprocess.
    Phase-3 goal: one package entrypoint for deployment and reproducible benchmarking.
    """

    def __init__(self, args, checkpoint_path: str, device: Optional[str] = None):
        self.args = args
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self._checkpoint_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(self._checkpoint_state, dict) and "state_dict" in self._checkpoint_state:
            self._checkpoint_state = self._checkpoint_state["state_dict"]

    def _build_model(self, args):
        in_channels = getattr(args, "in_channels", None)
        if in_channels is None:
            raise ValueError("`args.in_channels` is required unless using lazy init through `predict_delta`.")
        if args.model_type == "Net_coor_two_stage":
            return Net_coor_two_stage(in_channels, args)
        if args.model_type == "Net_coor_torsion":
            return Net_coor_torsion(in_channels, args)
        return Net_coor(in_channels, args)

    def _lazy_init_model(self, data):
        if self.model is not None:
            return
        if getattr(self.args, "in_channels", None) is None:
            self.args.in_channels = int(data.x.size(1))
        self.model = self._build_model(self.args).to(self.device)
        self.model.load_state_dict(self._checkpoint_state)
        self.model.eval()

    @staticmethod
    def _edge_attr(data, use_alpha_channel=False):
        edge_attr = data.dist.float()
        if use_alpha_channel and hasattr(data, "alpha"):
            edge_attr = torch.cat([edge_attr, data.alpha.float()], dim=1)
        return edge_attr

    @torch.no_grad()
    def predict_delta(self, data):
        self._lazy_init_model(data)
        data = data.to(self.device)
        edge_attr = self._edge_attr(data, use_alpha_channel=getattr(self.args, "use_alpha_channel", False))

        if self.args.model_type == "Net_coor_two_stage":
            pred_all, rank_logit = self.model(
                data.x.float(),
                data.edge_index,
                edge_attr,
                data.batch if hasattr(data, "batch") else None,
            )
            pred = pred_all[data.flexible_idx.bool()]
            return {
                "delta": pred,
                "rank_prob": torch.sigmoid(rank_logit),
            }

        if self.args.model_type == "Net_coor_torsion":
            pred_all, torsion_node = self.model(data.x.float(), data.edge_index, edge_attr)
            pred = pred_all[data.flexible_idx.bool()]
            return {
                "delta": pred,
                "torsion_node": torsion_node,
            }

        pred = self.model(data.x.float(), data.edge_index, edge_attr)[data.flexible_idx.bool()]
        return {"delta": pred}
