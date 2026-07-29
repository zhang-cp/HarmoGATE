import argparse
from pathlib import Path

import torch

from dataset import DRUG_DICT, PROT_DICT
from harmogate_model import HarmoGATE


VARIANTS = {
    "full",
    "without_graph",
    "without_cross_attention",
    "without_gated_fusion",
    "without_modal_consistency",
    "without_multitask_dti",
}


def build_model(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return HarmoGATE(
        drug_vocab_size=len(DRUG_DICT) + 1,
        protein_vocab_size=len(PROT_DICT) + 1,
        graph_in_dim=26,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        dropout=0.2,
        use_graph=variant != "without_graph",
        use_cross_attention=variant != "without_cross_attention",
        use_gated_fusion=variant != "without_gated_fusion",
        use_transformer=True,
    )


def load_model(checkpoint_path, variant="full", device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = build_model(variant).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_missing = set()
    if variant == "full":
        allowed_missing = {
            "simple_fusion_norm.weight",
            "simple_fusion_norm.bias",
        }

    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing.issubset(allowed_missing) or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the released model definition. "
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
