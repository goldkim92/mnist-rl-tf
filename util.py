import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance

'''
np2pil input img shape: [28,28,1]
pil3np output img shpae: [28,28,1]
'''
np2pil = lambda img: Image.fromarray((img.squeeze(2)*255).astype(np.uint8))
pil2np = lambda img: np.expand_dims((np.array(img) / 255.), axis=2)
pil_rotate = lambda img, angle: img.rotate(angle)
pil_blur = lambda img, radius: img.filter(ImageFilter.GaussianBlur(radius))
pil_sharpen = lambda img, radius: img.filter(ImageFilter.UnsharpMask(radius))


def np_rotate(img, angle):
    img = np2pil(img)
    img = pil_rotate(img, angle)
    img = pil2np(img)
    return img
    

def np_sharpen(img, radius):
    img = np2pil(img)
    img = pil_sharpen(img, radius)
    img = pil2np(img)
    return img


def random_degrade(img, angle=None, radius=None):
    '''
    img: 28*28*1 with each value ranged in [0,1]
    '''
    if angle==None and radius==None:
        angle = np.random.randint(-80,80)
#         radius = np.random.uniform(1.,2.)
    
    img = np2pil(img)
    img = pil_rotate(img, angle)
#     img = pil_blur(img, radius)
    img = pil2np(img)
    return img


def make_grid(batch, nrow=8, padding=2):
    '''
    batch: size = [batch_size, 28, 28 ,1]
    nrow: Number of images displayed in each row of the grid
    '''
    batch = batch.squeeze(axis=3)
    ds = batch.shape[1] # data_size
    ncol = np.ceil(batch.shape[0]/nrow).astype(np.int)
    grid = np.ones([(ds+padding)*ncol-padding, (ds+padding)*nrow-padding])
    
    for i in range(batch.shape[0]):
        row_idx, col_idx = i%nrow, i//nrow
        grid[col_idx*(padding+ds):col_idx*(padding+ds) + ds, 
             row_idx*(padding+ds):row_idx*(padding+ds) + ds] = batch[i]
    
    return grid 
    

def save_batch_fig(fname, batch_grid, img_width, angles):
    '''
    batch_grid: output of `make_grid` function
    img_width: width of original image
    angles: list of rotated angles
    '''
    xticks_position = np.arange(img_width//2, batch_grid.shape[1], img_width+2)
    angles = [f'{angle:.03f}' for angle in angles]

    fig = plt.figure(figsize=(12,3))
    plt.imshow(batch_grid, cmap='gray')
    plt.xticks(xticks_position, angles)
    plt.yticks([])
    for direction in ['bottom','left','top','right']:
        plt.gca().spines[direction].set_visible(False)
    plt.savefig(fname)
    plt.close(fig)

