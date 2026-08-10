import torch.nn as nn
from .lora import LoRALinear


def _get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model

    for p in parts[:-1]:
        parent = getattr(parent, p)

    return parent, parts[-1]


# def apply_lora_to_ghn(
#     model,
#     r=8,
#     alpha=16,
#     dropout=0.05,
#     target_keywords=None,
#     skip_keywords=None,
# ):
#     if target_keywords is None:
#         target_keywords = [
#             "gnn",
#             "attn",
#             "mlp",
#             "proj",
#             "fc",
#             "linear",
#             "decoder"
#         ]
#
#     if skip_keywords is None:
#         skip_keywords = [
#             "class_layer_predictor",
#             "bias_class",
#         ]
#
#     replaced = []
#
#     for name, module in list(model.named_modules()):
#         if not isinstance(module, nn.Linear):
#             continue
#
#         lname = name.lower()
#
#         if any(k.lower() in lname for k in skip_keywords):
#             continue
#
#         if not any(k.lower() in lname for k in target_keywords):
#             continue
#
#         parent, child_name = _get_parent_module(model, name)
#         old_linear = getattr(parent, child_name)
#
#         setattr(
#             parent,
#             child_name,
#             LoRALinear(
#                 old_linear,
#                 r=r,
#                 alpha=alpha,
#                 dropout=dropout,
#             ),
#         )
#
#         replaced.append(name)
#
#     return model, replaced

def apply_lora_to_ghn(
    model,
    r=8,
    alpha=16,
    dropout=0.05,
    target_keywords=None,
    skip_keywords=None,
):
    if target_keywords is None:
        target_keywords = [
            "attn.to_qkv",
            "attn.to_out",
        ]

    if skip_keywords is None:
        skip_keywords = [
            "class_layer_predictor",
            "bias_class",
        ]

    replaced = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        lname = name.lower()

        if any(k.lower() in lname for k in skip_keywords):
            continue

        # Ensure that only GNN attention layers are selected
        if not lname.startswith("gnn."):
            continue

        if not any(k.lower() in lname for k in target_keywords):
            continue

        parent, child_name = _get_parent_module(model, name)
        old_linear = getattr(parent, child_name)

        setattr(
            parent,
            child_name,
            LoRALinear(
                old_linear,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )

        replaced.append(name)

    return model, replaced


def freeze_except_lora_and_classifier(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True

        if "decoder.class_layer_predictor" in name:
            param.requires_grad = True

        if "bias_class" in name:
            param.requires_grad = True

    return model


def print_trainable_params(model, log_fn=print):
    trainable = 0
    total = 0

    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    pct = 100 * trainable / total if total > 0 else 0

    log_fn(f"Trainable params: {trainable:,}")
    log_fn(f"Total params: {total:,}")
    log_fn(f"Trainable %: {pct:.4f}%")

    for name, p in model.named_parameters():
        if p.requires_grad:
            log_fn(f"TRAINABLE: {name} {tuple(p.shape)}")

    return trainable, total, pct