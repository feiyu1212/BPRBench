import timm
import json
import os
from typing import Union
from timm.layers import set_layer_config
from timm.models import model_entrypoint
from timm.layers import SwiGLUPacked
import torch


def load_cfg_from_json(json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_conf_from_json(json_file: Union[str, os.PathLike], model_id: str):
    hf_config = load_cfg_from_json(json_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args


def create_empty_vir2(**kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    kwargs.setdefault("mlp_layer", SwiGLUPacked)
    kwargs.setdefault("act_layer", torch.nn.SiLU)
    pretrained_cfg = {
        "tag": "virchow_v2",
        "custom_load": False,
        "input_size": [
        3,
        224,
        224
        ],
        "fixed_input_size": False,
        "interpolation": "bicubic",
        "crop_pct": 1.0,
        "crop_mode": "center",
        "mean": [
        0.485, 
        0.456, 
        0.406
        ],
        "std": [
        0.229, 
        0.224, 
        0.225
        ],
        "num_classes": 0,
        "pool_size": None,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        "license": "MIT"
    }
    model_name = "vit_huge_patch14_224"
    model_args = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "reg_tokens": 4,
        "mlp_ratio": 5.3375,
        "global_pool": "",
        "dynamic_img_size": True
    }
    for k, v in model_args.items():
        kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config():
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs,
        )
    return model