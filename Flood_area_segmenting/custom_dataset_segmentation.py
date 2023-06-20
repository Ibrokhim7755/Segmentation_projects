from torch.utils.data import Dataset, DataLoader, random_split

"""
This class function is used to pull the dataset from the directory and defines parameter,
transformation, albumentation.
we have three method in this code to call the dataset:
def __init__():
initialize the root(path) and transformation

def __len__():
defines a number of images in the dataset

def __getitim__(): 
defines the items and label 

Parameters :
   A = albumentation
  im = Image
  gt = Ground Truth

im_path_jpg = Image path for Jpg extensions
im_mask_png = Mask path for png
    
"""



class Segmentations(Dataset):
    def __init__(self, root, transformations=None):
        self.transformations = transformations
        self.tensorize = T.Compose([T.ToTensor()])
        self.im_path_jpg = sorted(glob(f"{root}/images/*"))
        self.im_mask_png = sorted(glob(f"{root}/masks/*"))

        self.total_ims = len(self.im_path_jpg)
        self.total_gts = len(self.im_mask_png)

        assert self.total_ims == self.total_gts
        print(f'There are {self.total_ims} images and {self.total_gts} masks in the dataset')

    def __len__(self):
        return len(self.im_path_jpg)

    def __getitem__(self, idx):
        im = np.array(Image.open(self.im_path_jpg[idx]).convert('RGB'))
        gt = np.array(Image.open(self.im_mask_png[idx]))

        if self.transformations is not None:
            transformed = self.transformations(image=im, mask=gt)
            im, gt = transformed['image'], transformed['mask']

        return self.tensorize(im), torch.tensor(gt > 128).long()


def get_transformation(size):
    tfs = A.Compose([A.Resize(size, size)])
    return tfs

tfs = get_transformation(256)
ds = Segmentations(root='/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation/flood', transformations=tfs)
print(type(ds[0][0]))
print(ds[0][1].shape)