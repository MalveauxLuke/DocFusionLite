from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import yaml


# dataclasses

@dataclass
class ModelConfig:
    mode: Literal["3.1", "3.3"] = "3.1"
    d_model: int = 768
    d_region: int = 256
    d_patch: int = 768
    n_heads: int = 8
    num_fusion_layers: int = 0
    num_encoder_layers: int = 2
    num_labels: int = 16
    dropout: float = 0.1
    use_gate_stem: bool = False
    use_gate_fusion_layer: bool = False
    use_region_ffn: bool = True
    text_model_name: str = "microsoft/deberta-v3-base"
    vision_model_name: str = "microsoft/dit-base"


@dataclass
class TrainingConfig:
    batch_size: int = 2
    lr: float = 3e-5
    weight_decay: float = 0.01
    num_steps: int = 1000
    warmup_steps: int = 50


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


# loader

_SRC_DIR = Path(__file__).resolve().parent   # points to src/
_DEFAULT_CFG = _SRC_DIR / "configs" / "base.yaml"


def _update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _update_dataclass(cur, v)
        else:
            setattr(obj, k, v)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load YAML into Config.
    - If path is None -> src/configs/base.yaml
    - If path is relative -> interpreted relative to src/
    """
    cfg = Config()

    if path is None:
        yaml_path = _DEFAULT_CFG
    else:
        yaml_path = Path(path)
        if not yaml_path.is_absolute():
            yaml_path = _SRC_DIR / yaml_path

    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    _update_dataclass(cfg, raw)
    return cfg
