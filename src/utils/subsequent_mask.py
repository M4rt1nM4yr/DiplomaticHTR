import numpy as np
import torch

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = ~(torch.from_numpy(subsequent_mask) == 0).squeeze(0)
    subsequent_mask = torch.where(subsequent_mask,torch.ones(()),torch.zeros(()))
    return subsequent_mask.bool()

if __name__ == "__main__":
    t = subsequent_mask(5)
    print(t.shape)
    print(t)