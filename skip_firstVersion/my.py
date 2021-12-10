import torch
import torch.nn as nn

import numpy as np

# img_resoluton = 256
# channel_max = 512
# channel_base_dict = {32:1024, 64:2048, 128:4096, 256:8192, 512:16384, 1024:32769}
# channel_base = channel_base_dict[img_resoluton]
# img_log2 = int(np.log2(img_resoluton))

# img_block = [2 ** i for i in range(img_log2, 2, -1)]
# print(f"img_log2={img_log2}")

# channel_dict = {res:min(channel_base // res, channel_max ) for res in img_block + [4]}
# print(f"channel_dict = {channel_dict}")

# for idx, res in enumerate(img_block):
#             in_channel = channel_dict[res] if res < img_resoluton else 0
#             tmp_channel = channel_dict[res] 
#             out_channel = channel_dict[res // 2]
#             print(f"res = {res}, in_chan ={in_channel}, out_chan = {out_channel}")

# img_block = [2 ** i for i in range(img_log2, 2, -1)] + [4]
# print(img_block)
# for res in img_block[::-1][:-1]:
#     in_chan = channel_dict[res] * 2
#     out_chan = channel_dict[res * 2] if res < img_resoluton / 2 else channel_dict[res]
#     # print(f"res ={res}, channel_base = {channel_dict[res]}")
#     print(f"res = {res}, in_chan ={in_chan}, out_chan = {out_chan}")