import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "src"))

from models import build_model

for name in ["resnet18", "mobilenet_v3_large", "shufflenet_v2_x1_0", "efficientnet_b0", "swin_t", "resnet50"]:
    m, _ = build_model(name, 38, pretrained=False)
    print("===", name)
    if name.startswith("resnet"):
        print("   target = layer4[-1] ->", type(m.layer4[-1]).__name__)
    elif name in ("mobilenet_v3_large", "mobilenet_v3_small"):
        print("   target = features[-1] ->", type(m.features[-1]).__name__)
        print("   features[-1][0] ->", type(m.features[-1][0]).__name__)
    elif name == "shufflenet_v2_x1_0":
        print("   target = conv5 ->", type(m.conv5).__name__)
    elif name == "efficientnet_b0":
        print("   target = features[-1] ->", type(m.features[-1]).__name__)
    elif name.startswith("swin"):
        print("   target = features[-1][-1] ->", type(m.features[-1][-1]).__name__)
        print("   norm1 ->", type(m.features[-1][-1].norm1).__name__)
