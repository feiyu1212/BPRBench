from torch import nn

from .vir2_model import create_empty_vir2


class HfVisionModel(nn.Module):
    """Vision tower wrapper.

    For the Virchow2 variant the pretrained trunk is NOT downloaded here: the
    downstream factory always replaces ``self.trunk`` with an empty Virchow2
    ViT (``create_empty_vir2``) and the real weights are loaded from the user's
    Vir2_CLIP ``.pt`` checkpoint. Building the empty trunk directly avoids a
    wasteful (and possibly gated) HuggingFace download at model-construction
    time.
    """

    def __init__(self,
                 hf_model_name_or_path: str,
                 virchow2_special_configs=False,
                 ) -> None:
        super().__init__()
        if not virchow2_special_configs:
            from transformers import ViTModel
            self.trunk = ViTModel.from_pretrained(hf_model_name_or_path)
            self.is_virchow2 = False
        else:
            self.trunk = create_empty_vir2()
            self.is_virchow2 = True

    def forward(self, x):
        out = self.trunk(x)
        if self.is_virchow2:
            return out[:, 0]
        return out.pooler_output
