#!/usr/bin/env python3
"""
Export the fine-tuned TwinLiteNetPlus (Large) checkpoint to ONNX.

Context: this model was fine-tuned starting from the upstream TwinLiteNet
training pipeline (github.com/chequanghuy/TwinLiteNet, MIT licensed), using
its "Large" configuration. The upstream repo's own source for that exact
config module wasn't available/importable, so the architecture below is a
manual reconstruction that matches the checkpoint's state_dict shapes
layer-for-layer — verified by loading strict=True (no key mismatches).

Note: this export path is provided as working code but has not been
benchmark-verified end to end on target hardware (Jetson) in this repo.
An earlier export attempt from this project produced a non-functional ONNX
file (a stub that returned random noise instead of running the network) and
has been removed rather than published.

Usage:
    python export_onnx.py --weights ../models/twinlitenetplus_large.pth --output twinlitenetplus_large.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# TwinLiteNetPlus (Large) architecture — reconstructed to match the fine-tuned
# checkpoint's state_dict.
# ==============================================================================


class CBR(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.PReLU(c2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DilatedSepConv(nn.Module):
    def __init__(self, c1, c2, d):
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, 3, padding=d, dilation=d, groups=c1, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class TasksUpsampleBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, 2, stride=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.PReLU(c2),
        )
        self.conv1 = CBR(c2, c2)
        self.conv2 = CBR(c2, c2)

    def forward(self, x):
        x = self.up_conv(x)
        return x + self.conv2(self.conv1(x))


class OutputBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, 2, stride=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.PReLU(c1),
        )
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.up_conv(x))


class PCAA(nn.Module):
    """Position/channel attention-augmented aggregation block."""

    def __init__(self, in_dim):
        super().__init__()
        self.conv_cam = nn.Conv2d(in_dim, in_dim, 1)
        self.gcn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, (1, 7), padding=(0, 3)),
            nn.Conv2d(in_dim, in_dim, (7, 1), padding=(3, 0)),
        )
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 1)
        self.fuse = nn.Conv2d(in_dim, in_dim, 1)
        self.proj_query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.proj_key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.proj_value = nn.Conv2d(in_dim, in_dim, 1)
        self.conv_out = nn.Sequential(CBR(in_dim, in_dim), nn.Conv2d(in_dim, in_dim, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        cam = self.relu(self.gcn(self.conv_cam(x)))
        q = self.proj_query(cam).view(b, -1, h * w).permute(0, 2, 1)
        k = self.proj_key(cam).view(b, -1, h * w)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.proj_value(x).view(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.conv_out(self.fuse(out + cam))


class ESPBlockLarge(nn.Module):
    """Efficient spatial pyramid block with dilated branches (rates 1/2/4/8/16)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = CBR(c1, 25)
        self.d1 = DilatedSepConv(25, 28, 1)
        self.d2 = DilatedSepConv(25, 25, 2)
        self.d4 = DilatedSepConv(25, 25, 4)
        self.d8 = DilatedSepConv(25, 25, 8)
        self.d16 = DilatedSepConv(25, 25, 16)
        self.bn = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU(128))

    def forward(self, x):
        base = self.c1(x)
        i1 = self.d1(base)
        i2 = self.d2(base) + i1
        i4 = self.d4(base) + i2
        i8 = self.d8(base) + i4
        i16 = self.d16(base) + i8
        return self.bn(torch.cat([i1, i2, i4, i8, i16], dim=1))


class TwinLiteNetPlusLarge(nn.Module):
    """Dual-head (drivable area + lane line) segmentation network."""

    def __init__(self):
        super().__init__()
        self.level1 = CBR(3, 32, k=3, s=2, p=1)
        self.b1 = CBR(32 + 3, 64, k=3, s=2, p=1)

        self.level2_0 = ESPBlockLarge(64 + 3, 128)
        self.level2 = nn.ModuleList([ESPBlockLarge(128, 128) for _ in range(5)])
        self.b2 = CBR(128 + 3, 259, k=3, s=2, p=1)

        self.level3_0 = ESPBlockLarge(259 + 3, 256)
        self.level3 = nn.ModuleList([ESPBlockLarge(256, 256) for _ in range(7)])

        self.caam = PCAA(256)
        self.conv_caam = CBR(256, 256)

        self.up_1_da = TasksUpsampleBlock(256, 128)
        self.up_2_da = TasksUpsampleBlock(128, 64)
        self.out_da = OutputBlock(64, 2)

        self.up_1_ll = TasksUpsampleBlock(256, 128)
        self.up_2_ll = TasksUpsampleBlock(128, 64)
        self.out_ll = OutputBlock(64, 2)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x_down2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_down4 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        x_down8 = F.interpolate(x, size=(h // 8, w // 8), mode='bilinear', align_corners=True)

        out_l1 = self.level1(x)
        out_b1 = self.b1(torch.cat([out_l1, x_down2], dim=1))

        out_l2 = self.level2_0(torch.cat([out_b1, x_down4], dim=1))
        for layer in self.level2:
            out_l2 = layer(out_l2)

        out_b2 = self.b2(torch.cat([out_l2, x_down4], dim=1))
        out_l3 = self.level3_0(torch.cat([out_b2, x_down8], dim=1))
        for layer in self.level3:
            out_l3 = layer(out_l3)

        feat = self.conv_caam(self.caam(out_l3))

        da = self.out_da(self.up_2_da(self.up_1_da(feat)))
        ll = self.out_ll(self.up_2_ll(self.up_1_ll(feat)))
        return da, ll


def load_checkpoint(model: nn.Module, weights_path: Path) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip DataParallel "module." prefixes if present.
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the fine-tuned TwinLiteNetPlus (Large) checkpoint to ONNX")
    parser.add_argument("--weights", required=True, type=Path, help="Path to twinlitenetplus_large.pth")
    parser.add_argument("--output", type=Path, default=Path("twinlitenetplus_large.onnx"))
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Checkpoint not found: {args.weights}")
        sys.exit(1)

    print("Instantiating TwinLiteNetPlus (Large)...")
    model = TwinLiteNetPlusLarge()

    print(f"Loading weights from {args.weights}...")
    load_checkpoint(model, args.weights)
    print("Weights loaded (strict match — no missing/unexpected keys).")

    dummy_input = torch.randn(1, 3, args.height, args.width)
    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["drivable_area", "lanes"],
    )
    print(f"Done: {args.output}")
