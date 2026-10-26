'''
LastEditors: Songlin Zhu zhusonglin@nuaa.edu.cn
Date: 2026-10-26 14:20:50
LastEditTime: 2026-10-26 14:26:33
FilePath: \\basisHyperVectorGenerate\\seedgen.py
'''
r'''
Date: 2025-07-25 15:30:31
LastEditors: zhusonglin zhusonglin@nuaa.com
LastEditTime: 2025-11-27 15:45:50
FilePath: \\shtHDC\\seedgen.py
'''

import math
import torch

r'''
seedgen(dim, step, polar)
use math method to generate seed, and then exchange to string
for example:
    out = seedgen(8, 4, 0)
    out = [0, 0, 0, 0, 1, 1, 1, 1]

    out = seedgen(8, 2, 1)
    out = [1, 1, 0, 0, 1, 1, 0, 0]
'''

def seedgen(dim, step, polar = 0):
    if polar not in (0, 1):
        raise ValueError("polar must be 0 or 1")

    if not (dim > 0 and (dim & (dim - 1)) == 0):
        raise ValueError("dim must be an integer power of 2")

    if not (step > 0 and (step & (step - 1)) == 0):
        raise ValueError("step must be an integer power of 2")

    if dim <= step:
        raise ValueError("step must be smaller than dim")

    x = 2 ** (dim) - 1
    
    if polar == 0:
        s = int(x // (2 ** step + 1))
    else:
        s = int(2 ** dim - 1) - int(x // (2 ** step + 1))
        # s = int(x / (1 / (2 ** step) + 1))

    s_list = int_to_bin_list(s, dim)      # Convert to strings

    return s_list
        
def int_to_bin_list(num, total_len=None):
    f'''
    Converts integers into binary lists
    num: Enter an integer
    total_len: Binary total length, with a leading zero if specified
    Return: Binary list [0,1,...]
    '''
    if num < 0:
        raise ValueError("Only non-negative integers are supported")
    # Convert to binary strings
    bin_str = bin(num)[2:]  # bin() return '0b1011'，remove '0b'
    # Supplement 0 to total_len
    if total_len is not None:
        bin_str = bin_str.zfill(total_len)
    # Convert to list
    return [int(c) for c in bin_str]


def seedgen_all(dim, polar = 0, dataType = "bipolar", dtype = torch.float64, device = "cpu"):
    # print(dim)
    if not (dim > 0 and (dim & (dim - 1)) == 0):
        raise ValueError("dim must be an integer power of 2")
    numMAX = int(math.log2(dim))
    listRT = torch.zeros([numMAX + 1, dim], dtype=torch.float)
    for n in range(0, numMAX):
        # print(numMAX)
        listRT[(numMAX + 1) - (n + 1)] = torch.tensor(seedgen(dim, 2**n, polar = 0))
    # print(listRT)
    # print(torch.where(listRT.bool(), +1, -1))
    if dataType == "binary":
        return (listRT, (numMAX, dim))
    if dataType == "bipolar":
        return (torch.where(listRT.bool(), +1, -1).to(dtype=dtype, device=device), (numMAX, dim))