from skimage.draw import circle
import numpy as np
import torch
import torch.nn.functional as F
margin = 5
img = np.zeros((margin, margin), dtype=np.uint8)

rr, cc = circle(margin/2, margin/2, margin/2+1, shape=(margin, margin))
img[rr, cc] = 1
print(img)
fixed_depth = 6
#img = np.repeat(np.expand_dims(np.expand_dims(img, 0), 0), fixed_depth, 0) #[6, 1, 10, 10]
print(img, img.shape)

label = np.array([[1]*10, [2]*10, [3]*10,[4]*10, [5]*10, [6]*10, [7]*10, [8]*10, [9]*10, [0]*10], dtype=np.int32)
#background
back = np.zeros((fixed_depth, label.shape[0], label.shape[1]))
print(label, label.shape, back, back.shape) #[6, 10, 10]

object_list = []

bincount = np.bincount(label.flatten())  # count number of occurences of each value non-negative ints
print(bincount)
min_fragment = 5
pixels = np.where(bincount > min_fragment)[0]  # M^K (k, 1)
print(pixels)
if len(pixels) > fixed_depth:
    pixels = pixels[:fixed_depth]
    print("Not all objects fits in fixed depth", RuntimeWarning)
#
for l, v in enumerate(pixels):
    back[l, label == v] = 1.
    object_list.append(np.array(range(l + 1)))
print(back, object_list) #object id in the original
sel = torch.from_numpy(img).float()

labels = torch.from_numpy(back).float()
labels = torch.unsqueeze(labels, dim=0)

sel = torch.unsqueeze(torch.stack([sel]*fixed_depth, dim=0), dim=1) #

print(labels.shape, sel.shape) #[1, 6, 10, 10] 6 is in channel , [6, 1, 10, 10], 6 is out channel
#exit(0)
#
print(labels.data[0,0], sel.data[0,0])
# labels = torch.from_numpy(back).float().to(device)
masks = F.conv2d(labels, sel, groups=fixed_depth, padding=margin // 2)
print(masks, masks.shape) #[1, 6, 5, 5]

# after convolution we get
#
# masks[masks > 0] = 1.
# masks[labels > 0] = 2.
# masks[:, 0, :, :] = 1.
