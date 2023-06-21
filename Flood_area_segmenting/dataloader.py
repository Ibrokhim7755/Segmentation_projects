'''

This function defines the root, transformation, batchsize and split the data into train validation and test.

Parameters: 
tr_len    = lenth of train set
val_len   = lenth of validation set
test_len  = lenth of test set

tr_dl     = train dataloader 
val_dl    = validation dataloader
test_dl   = test dataloader.

'''

def get_dl(root, transformations, bs, split = [0.7, 0.15, 0.15]):
    
    assert sum(split) == 1
    # Get dataset
    ds = Segmentations(root = "/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation/flood", transformations = tfs)
    
    tr_len = int(split[0] * len(ds))
    val_len = int(split[1] * len(ds))
    test_len = len(ds) - (tr_len + val_len)
    # Data split
    tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_len, val_len, test_len])
    
    print(f"There are {len(tr_ds)} images in the trainset")
    print(f"There are {len(val_ds)} images in the validation set")
    print(f"There are {len(test_ds)} images in the test set")
    
    tr_dl = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True)
    val_dl = DataLoader(dataset = val_ds, batch_size = bs, shuffle = False)
    test_dl = DataLoader(dataset = test_ds, batch_size = bs, shuffle = False)
    
    return tr_dl, val_dl, test_dl 
    
tr_dl, val_dl, test_dl = get_dl(root = "data", bs = 64, transformations = tfs) 