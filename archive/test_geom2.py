import numpy as np
from scipy import stats
from skimage import measure

def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = int(np.atleast_1d(info_region[0])[0]) - 1
    
    return no_region

def test():
    img = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    res = fea_geom(img)
    print("Success!", res)
test()
