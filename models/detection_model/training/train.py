from pathlib import Path

from rfdetr import RFDETRSmall

_PKG_DIR = Path(__file__).resolve().parents[1]

def main():
    model = RFDETRSmall()

    model.train(
        dataset_dir=str(_PKG_DIR / "dataset"),

        wandb=True,
        project="smartphone-detector",
        run="rfdetr-small-v1",

        resume=str(_PKG_DIR / "model" / "checkpoint.pth"),
        
        epochs=50,           
        batch_size=2,        
        grad_accum_steps=8,  
        imgsz=512            
    )

if __name__ == "__main__":
    main()