from rfdetr import RFDETRSmall

def main():
    model = RFDETRSmall()

    model.train(
        dataset_dir="./dataset", 

        wandb=True,
        project="smartphone-detector", 
        run="rfdetr-small-v1",
        
        resume="./output/checkpoint.pth", 
        
        epochs=50,           
        batch_size=2,        
        grad_accum_steps=8,  
        imgsz=512            
    )

if __name__ == "__main__":
    main()