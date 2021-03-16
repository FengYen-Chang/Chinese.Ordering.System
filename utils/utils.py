import numpy as np

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].shape[0]for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = np.pad(
                batch, (0, max_len-batch.shape[0]), 
                mode="constant", constant_values=0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = np.pad(
                batch, ((0, 0), (0, max_len-batch.shape[0])), 
                mode="constant", constant_values=0.0)
        out_list.append(one_batch_padded)
    out_padded = np.stack(out_list)
    return out_padded
