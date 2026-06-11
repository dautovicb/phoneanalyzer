from pathlib import Path

from rfdetr import RFDETRSmall

_PKG_DIR = Path(__file__).resolve().parents[1]

model = RFDETRSmall(pretrain_weights=str(_PKG_DIR / "model" / "checkpoint_best_total.pth"))

model.export()