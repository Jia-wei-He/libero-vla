"""LIBERO-VLA with Qwen2.5-VL multimodal context and a Flow Matching head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .action_head import FlowMatchingActionHead
from .qwen_vlm_backbone import QwenVLMBackbone


@dataclass
class VLAConfig:
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_dtype: str = "bfloat16"
    vlm_attn_implementation: str = "sdpa"
    vlm_freeze_vision: bool = True
    vlm_freeze_language: bool = True
    vlm_freeze_merger: bool = True

    processor_min_pixels: int | None = None
    processor_max_pixels: int | None = None
    processor_padding: bool | str = True
    processor_truncation: bool = False
    processor_max_length: int | None = None
    processor_add_generation_prompt: bool = True

    action_dim: int = 7
    chunk_size: int = 10
    num_flow_steps: int = 10
    action_hidden_dim: int = 512
    action_num_layers: int = 4
    action_num_heads: int = 8

    use_wrist: bool = False
    use_proprio: bool = False
    proprio_dim: int = 5

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VLAConfig":
        m = cfg["model"]
        d = cfg.get("data", {})
        v = m["vlm"]
        p = v.get("processor", {})
        freeze = v.get("freeze", {})
        a = m["action_head"]
        return cls(
            vlm_name=v["name"],
            vlm_dtype=v.get("dtype", "bfloat16"),
            vlm_attn_implementation=v.get("attn_implementation", "sdpa"),
            vlm_freeze_vision=bool(freeze.get("vision", True)),
            vlm_freeze_language=bool(freeze.get("language", True)),
            vlm_freeze_merger=bool(freeze.get("merger", True)),
            processor_min_pixels=p.get("min_pixels"),
            processor_max_pixels=p.get("max_pixels"),
            processor_padding=p.get("padding", True),
            processor_truncation=bool(p.get("truncation", False)),
            processor_max_length=p.get("max_length"),
            processor_add_generation_prompt=bool(
                p.get("add_generation_prompt", True)
            ),
            action_dim=int(a["action_dim"]),
            chunk_size=int(a["chunk_size"]),
            num_flow_steps=int(a["num_flow_steps"]),
            action_hidden_dim=int(a["hidden_dim"]),
            action_num_layers=int(a["num_layers"]),
            action_num_heads=int(a["num_heads"]),
            use_wrist=bool(d.get("use_wrist", False)),
            use_proprio=bool(d.get("use_proprio", False)),
            proprio_dim=int(d.get("proprio_dim", 5)),
        )


class LIBEROVLA(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        self.vlm = QwenVLMBackbone(
            model_name=config.vlm_name,
            min_pixels=config.processor_min_pixels,
            max_pixels=config.processor_max_pixels,
            dtype=config.vlm_dtype,
            attn_implementation=config.vlm_attn_implementation,
            freeze_vision=config.vlm_freeze_vision,
            freeze_language=config.vlm_freeze_language,
            freeze_merger=config.vlm_freeze_merger,
        )
        vlm_dim = self.vlm.hidden_size

        if config.use_proprio:
            self.proprio_mlp = nn.Sequential(
                nn.Linear(config.proprio_dim, vlm_dim),
                nn.SiLU(),
                nn.Linear(vlm_dim, vlm_dim),
            )

        self.action_head = FlowMatchingActionHead(
            vlm_dim=vlm_dim,
            action_dim=config.action_dim,
            num_queries=config.chunk_size,
            hidden_dim=config.action_hidden_dim,
            num_layers=config.action_num_layers,
            num_heads=config.action_num_heads,
            num_flow_steps=config.num_flow_steps,
        )

    @property
    def processor(self):
        return self.vlm.processor

    def encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action-head context and its padding mask.

        Qwen performs all image patching, image-token placement and multimodal
        fusion. The only extra prefix kept by LIBERO-VLA is one proprio token.
        """
        hidden = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        valid = attention_mask.bool()

        if self.config.use_proprio:
            if proprio is None:
                raise ValueError("use_proprio=True but proprio=None")
            mlp_dtype = next(self.proprio_mlp.parameters()).dtype
            prop_tok = self.proprio_mlp(proprio.to(mlp_dtype))
            prop_tok = prop_tok.to(hidden.dtype).unsqueeze(1)
            hidden = torch.cat([prop_tok, hidden], dim=1)
            prop_valid = torch.ones(
                (valid.size(0), 1), dtype=torch.bool, device=valid.device
            )
            valid = torch.cat([prop_valid, valid], dim=1)

        action_head_dtype = next(self.action_head.parameters()).dtype
        hidden = hidden.to(action_head_dtype)
        return hidden, ~valid

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        actions: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, key_padding_mask = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            proprio=proprio,
        )
        if actions is not None:
            loss = self.action_head.loss(
                actions, hidden, ctx_key_padding_mask=key_padding_mask
            )
            return {
                "loss": loss,
                "hidden": hidden,
                "key_padding_mask": key_padding_mask,
            }
        pred = self.action_head.sample(
            hidden, ctx_key_padding_mask=key_padding_mask
        )
        return {
            "actions": pred,
            "hidden": hidden,
            "key_padding_mask": key_padding_mask,
        }

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
