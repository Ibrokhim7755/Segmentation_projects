import matplotlib.pyplot as plt

'''
This code plots the learning curves of the results:

Parameters:  

tr_loss  = Train loss
tr_pa    = Train Pixel accuracy
tr_iuo   = train intersection on units
val_loos = validation loss
val_pa   = validation pixel accuracy
val_iuo  = validation intersection on units
'''


def plot_results(his):
    tr_loss = [loss.item() for loss in his['tr_loss']]
    tr_pa = his["tr_pa"]
    tr_iou = his["tr_iou"]
    val_loss = his["val_loss"]
    val_pa = his["val_pa"]
    val_iou = his["val_iou"]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(tr_loss, label="Train_Loss")
    plt.plot(val_loss, label="Validation_Loss")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(tr_pa, label="Train_PA")
    plt.plot(val_pa, label="Validation_PA")
    plt.title("Pixel Accuracy (PA)")
    plt.ylabel("PA")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(tr_iou, label="Train_IoU")
    plt.plot(val_iou, label="Validation_IoU")
    plt.title("Intersection over Union (IoU)")
    plt.ylabel("IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    plot_results(his)