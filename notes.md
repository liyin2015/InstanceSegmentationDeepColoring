If you want to apply a per-channel convolution then your out-channel should be the same as your in-channel. This is expected, considering each of your input channels creates a separate output channel that it corresponds to.

In short, this will work

import torch
import torch.nn.functional as F

filters = torch.autograd.Variable(torch.randn(3,1,3,3))
inputs = torch.autograd.Variable(torch.randn(1,3,10,10))
out = F.conv2d(inputs, filters, padding=1, groups=3)
whereas, filters of size (2, 1, 3, 3) or (1, 1, 3, 3) will not work.

Additionally, you can also make your out-channel a multiple of in-channel. This works for instances where you want to have multiple convolution filters for each input channel.

However, This only makes sense if it is a multiple. If not, then pytorch falls back to its closest multiple, a number less than what you specified. This is once again expected behavior. For example a filter of size (4, 1, 3, 3) or (5, 1, 3, 3), will result in an out-channel of size 3.