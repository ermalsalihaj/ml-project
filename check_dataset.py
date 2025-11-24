import os

img_dir = "data/DUTS/images"
mask_dir = "data/DUTS/masks"

print("Images:", len(os.listdir(img_dir)))
print("Masks:", len(os.listdir(mask_dir)))
