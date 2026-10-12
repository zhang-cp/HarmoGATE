import argparse
from pathlib import Path

import torch

from dataset import DRUG_DICT, PROT_DICT
from harmogate_model import HarmoGATE
from harmogate_model_legacy import LegacyHarmoGATE


CURRENT_VARIANTS = {
    "full": {},
    "dta_only": {},
    "dti_only": {},
    "without_modal_consistency": {},
    "without_graph": {"use_graph": False},
    "without_cross_attention": {"use_cross_attention": False},
    "without_residual_mlp": {"use_fusion_residual": False},
    "without_interaction_branch": {
        "use_cross_attention": False,
        "use_interaction_branch": False,
    },
    "global_attention_fusion": {
        "use_gated_fusion": False,
        "fusion_mode": "global_attention",
    },
    "pure_global_attention_fusion": {
        "use_gated_fusion": False,
        "fusion_mode": "global_attention_residual",
    },
}

LEGACY_VARIANTS = {
    "legacy_full": {},
    "legacy_without_graph": {"use_graph": False},
    "legacy_without_cross_attention": {"use_cross_attention": False},
    "legacy_without_gated_fusion": {"use_gated_fusion": False},
}

VARIANTS = set(CURRENT_VARIANTS) | set(LEGACY_VARIANTS)


def build_model(variant):
    if variant in CURRENT_VARIANTS:
        return HarmoGATE(
            drug_vocab_size=len(DRUG_DICT) + 1,
            protein_vocab_size=len(PROT_DICT) + 1,
            graph_in_dim=26,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            dropout=0.2,
            use_jczs=False,
            use_transformer=True,
            **CURRENT_VARIANTS[variant],
        )
    if variant in LEGACY_VARIANTS:
        return LegacyHarmoGATE(
            drug_vocab_size=len(DRUG_DICT) + 1,
            protein_vocab_size=len(PROT_DICT) + 1,
            graph_in_dim=26,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            dropout=0.2,
            use_transformer=True,
            **LEGACY_VARIANTS[variant],
        )
    raise ValueError(f"Unknown variant: {variant}")


def read_state_dict(checkpoint_path, device):
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


def load_model(checkpoint_path, variant="full", device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    state_dict = read_state_dict(checkpoint_path, device)
    model = build_model(variant).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_missing = set()
    if variant.startswith("legacy_"):
        allowed_missing = {
            "simple_fusion_norm.weight",
            "simple_fusion_norm.bias",
        }

    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing.issubset(allowed_missing) or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the selected HarmoGATE variant. "
            f"Missing keys: {sorted(missing)}; "
            f"unexpected keys: {sorted(unexpected)}"
        )

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Load a HarmoGATE checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    load_model(args.checkpoint, args.variant, args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model variant: {args.variant}")


if __name__ == "__main__":
    main()
