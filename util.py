import numpy as np
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
        angle = np.random.randint(-60,60)
        radius = np.random.uniform(1.5,3.)
    
    img = np2pil(img)
    img = pil_rotate(img, angle)
    img = pil_blur(img, radius)
    img = pil2np(img)
    return img
