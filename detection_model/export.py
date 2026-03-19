from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="./output/checkpoint_best_total.pth")

model.export()