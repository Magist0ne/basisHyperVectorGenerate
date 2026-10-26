r'''
Date: 2025-07-14 08:58:34
LastEditors: zhusonglin zhusonglin@nuaa.com
LastEditTime: 2025-11-27 15:49:50
FilePath:
'''




import math
import random
import string

import uuid
import hashlib
from typing import Type, Union, Optional, Literal, Callable, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import torchhd.functional as functional

from torchhd.functional import get_vsa_tensor_class

from torchhd.types import VSAOptions

from itertools import chain, combinations


from typing import Type, Callable, Optional
from torch import LongTensor, FloatTensor, Tensor
from collections import deque
import warnings

from torchhd.tensors.base import VSATensor
from torchhd.tensors.bsc import BSCTensor
from torchhd.tensors.map import MAPTensor
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.fhrr import FHRRTensor, type_conversion as fhrr_type_conversion
from torchhd.tensors.bsbc import BSBCTensor
from torchhd.tensors.vtb import VTBTensor
from torchhd.tensors.mcr import MCRTensor
from torchhd.tensors.cgr import CGRTensor
from torchhd.types import VSAOptions



import seedgen
    
supported_dtypes: Set[torch.dtype] = {
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

class _baseTensorDictIns:
    def __init__(
        self, 
        dim,
        name = "",
        RecordCreateHistory: bool = True, # Record vector creation history
    ) -> None:
        
        if not (dim > 0 and (dim & (dim - 1)) == 0):
            raise ValueError(f"dimensions must be a series power of 2, but got {dim}")
        baseTensorNum = int(math.log(dim, 2))
        maxTensorNum = dim

        length = 12
        letters = string.ascii_letters + string.digits  # Includes uppercase/lowercase letters and digits
        rand_str = ''.join(random.choice(letters) for _ in range(length))
        self._name = name if name != "" else rand_str
        
        unique_id = uuid.uuid4().hex + self._name
        hashed_id = hashlib.sha256(unique_id.encode()).hexdigest()
        self.id = hashed_id
        
        # Vector creation count
        self.Count = 0 
        # Vector creation history enable flag
        self.RecordCreateHistory = RecordCreateHistory
        # If recording history, store it here
        self.__CreateList = []
        # self.__AllTensorList = []
        # for r in range(baseTensorNum + 1):
        #     combs = list(combinations(range(1, baseTensorNum + 1), r))
        #     self.__AllTensorList += combs
        # Pre-generate all vector table indices
        # self.__AllTensorList = list(chain.from_iterable(combinations(range(1, baseTensorNum + 1), r) for r in range(dim + 1)))
        self.__AllTensorList = [
            [0] if x == () else list(x)
            for x in chain.from_iterable(
                combinations(range(1, baseTensorNum + 1), r)
                for r in range(dim + 1)
            )   
        ]
        
        # print(self.__AllTensorList)
        # Vector generation count
        self.CountMax = len(self.__AllTensorList)

        if self.CountMax != maxTensorNum:
            raise ValueError(f"something is wrong in CountMax !")


    def _print_AllTensorList(self):
        print(self.__AllTensorList)
        print(len(self.__AllTensorList))

    def getID(self):
        return self.id

    def popTensor(self):
        # print(self.__AllTensorList)
        if not self.__AllTensorList:
            # print(self.__AllTensorList)
            raise BufferError("Tensor list is empty!")
        if self.Count >= self.CountMax:
            raise BufferError(f"The tensor is all used. Buffer is empty!")
        
        index = random.randint(0, len(self.__AllTensorList) - 1)
        # print(self.__AllTensorList)
        rtTensor = self.__AllTensorList.pop(index)
        # print(self.__AllTensorList)
        if self.RecordCreateHistory == True:
            self.__CreateList.append(rtTensor)
        self.Count += 1
        return rtTensor


    def popLastTensor(self):
        if not self.__AllTensorList:
            raise BufferError("Tensor list is empty!")
        if self.Count >= self.CountMax:
            raise BufferError(f"The tensor is all used. Buffer is empty!")

        rtTensor = self.__AllTensorList.pop()
        self.__CreateList.append(rtTensor)
        self.Count += 1
        return rtTensor


    def popFirstTensor(self):
        if not self.__AllTensorList:
            raise BufferError("Tensor list is empty!")
        if self.Count >= self.CountMax:
            raise BufferError(f"The tensor is all used. Buffer is empty!")

        rtTensor = self.__AllTensorList.pop(0)
        self.__CreateList.append(rtTensor)
        self.Count += 1
        return rtTensor

class baseTensorDictIns(_baseTensorDictIns):
    pass



# This class is a secondary wrapper around _baseTensorDictIns, mainly for binarization and integration with torchhd
class   basisTensor(MAPTensor):

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPTensor":
        
        if not (dimensions > 0 and (dimensions & (dimensions - 1)) == 0):
            raise ValueError(f"dimensions must be a series power of 2, but got {dimensions}")
        baseTensorNum = int(math.log(dimensions, 2))
        if num_vectors > dimensions:
            raise ValueError(f"The num_vectors must <= dimensions.")
        
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")
        
        if device == None:
            raise ValueError(f"a device is needed, such as cpu/xpu/cuda.")
        
        size = (num_vectors, dimensions)
        
        seedIndex = _baseTensorDictIns(dimensions)
        seedTensor = seedgen.seedgen_all(dimensions, polar = 0, dtype = dtype, device = device)
        # bioSeedTensor = cls._binary2bipolar(seedTensor)
        
        select = torch.empty(size, dtype=dtype, device=device)
        for i in range(0, num_vectors):
            # seedIndex._print_AllTensorList()
            index = seedIndex.popTensor()
            # print(f"index is {index}")
            select[i] = cls.groupIndex2Tensor(seedTensor, index, dtype=dtype, device=device, dform = "bipolar")

        return select.as_subclass(cls) 
        # size = (num_vectors, dimensions)
        # select = torch.empty(size, dtype=torch.bool, device=device)
        # select.bernoulli_(generator=generator)

        # result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
        # result.requires_grad = requires_grad
        # return result.as_subclass(cls)        


    

    # def __init__(
    #     self, 
    #     dim,
    #     name = "",
    #     dtype: torch.dtype = None,
    #     device: torch.device = None,
    #     dform: str = None,
    #     RecordCreateHistory: bool = False,
    # ) -> None: 
    #     if not (dim > 0 and (dim & (dim - 1)) == 0):
    #         raise ValueError(f"dimensions must be a series power of 2, but got {dim}")
        
    #     # input = torch.as_tensor(data, dtype=dtype, device=device)   
    #     if dtype not in supported_dtypes:
    #         options = ", ".join([str(x) for x in supported_dtypes])
    #         raise ValueError(f"{name} vectors must be one of dtype {options}.")
        
    #     if device == None:
    #         raise ValueError(f"a device is needed, such as cpu/xpu/cuda.")
        
    #     self.dtype = dtype
    #     self.device = device
        
    #     self.dim = dim       
    #     self.tensorDict = _baseTensorDictIns(dim)
    #     self.seedTensor = seedgen.seedgen_all(dim, polar=0)
    #     self.bioSeedTensor = self._binary2bipolar(self.seedTensor)
        
    #     if dform not in {"bipolar","binary"}:
    #         return ValueError(f"The dform need to be bipolar or binary.")
    #     self.dform = dform

    @classmethod
    def _bipolar2binary(cls, inputTensor: VSATensor, device = None):
        return torch.where(inputTensor > 0,  torch.tensor(1, device=device), torch.tensor(-1, device=device))
        
    @classmethod
    def _binary2bipolar(cls, inputTensor: VSATensor, device = None):
        return torch.where(inputTensor > 0.5,  torch.tensor(1, device=device), torch.tensor(0, device=device))
        

    # First convert to bipolar form, then use multiplication
    @classmethod
    def groupIndex2Tensor(cls, seedTensor, listIndex, dtype, device, dform = "binary"):
        # print(seedTensor)
        rtTensor = -1 * torch.ones((1, int(seedTensor[1][1])), dtype=torch.bool, device=device)
        # print(rtTensor)
        for i in listIndex:
            rtTensor = torch.mul(rtTensor, seedTensor[0][i])
        if dform == "binary":
            return cls._bipolar2binary(rtTensor)
        elif dform == "bipolar":
            return rtTensor
        else:
            raise ValueError(f"dform error")
        

    def level(
        num_vectors: int,
        dimensions: int,
        startTensor = None,
        endTensor = None,
        dtype = None,
        device = None,
        vsa: VSAOptions = "MAP",
        *,
        randomness: float = 0.0,
        requires_grad=False,
        **kwargs,
    ) -> VSATensor:
        """Creates a set of level correlated hypervectors.

        Implements level-hypervectors as an interpolation between random-hypervectors as described in `An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing <https://arxiv.org/abs/2205.07920>`_.
        The first and last hypervector in the generated set are quasi-orthogonal.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
            randomness (float, optional): r-value to interpolate between level at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.level(5, 6, "BSC")
            tensor([[ True,  True,  True,  True, False, False],
                    [ True,  True,  True,  True, False, False],
                    [False,  True,  True,  True,  True, False],
                    [False,  True,  True,  True,  True, False],
                    [False,  True,  True,  True,  True, False]])

            >>> torchhd.level(5, 6, "MAP")
            tensor([[ 1.,  1., -1.,  1., -1.,  1.],
                    [ 1.,  1.,  1.,  1., -1.,  1.],
                    [ 1.,  1.,  1.,  1., -1.,  1.],
                    [ 1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1., -1.,  1.,  1.,  1., -1.]])

            >>> torchhd.level(5, 6, "FHRR")
            tensor([[-0.996+0.079j,  0.447+0.894j, -0.840-0.541j, -0.999+0.020j, -0.742+0.669j, -0.999+0.042j],
                    [-0.886-0.462j,  0.447+0.894j, -0.840-0.541j, -0.999+0.020j, -0.742+0.669j, -0.886+0.462j],
                    [-0.886-0.462j,  0.447+0.894j, -0.146-0.989j, -0.999+0.020j, -0.350-0.936j, -0.886+0.462j],
                    [-0.886-0.462j,  0.507+0.861j, -0.146-0.989j, -0.999+0.020j, -0.350-0.936j, -0.886+0.462j],
                    [-0.886-0.462j,  0.507+0.861j, -0.146-0.989j, -0.611-0.791j, -0.350-0.936j, -0.886+0.462j]])

        """

        if startTensor == None or endTensor == None:
            raise ValueError(f"startTensor and endTensor is must needed!")
        
        vsa_tensor = get_vsa_tensor_class(vsa)

        # convert from normalized "randomness" variable r to number of orthogonal vectors sets "span"
        levels_per_span = (1 - randomness) * (num_vectors - 1) + randomness * 1
        # must be at least one to deal with the case that num_vectors is less than 2
        levels_per_span = max(levels_per_span, 1)
        span = (num_vectors - 1) / levels_per_span

        # generate the set of orthogonal vectors within the level vector set
        # span_hv = vsa_tensor.random(
        #     int(math.ceil(span + 1)),
        #     dimensions,
        #     **kwargs,
        # )
        span_hv = [startTensor, endTensor]

        # for each span within the set create a threshold vector
        # the threshold vector is used to interpolate between the
        # two random vector bounds of each span.
        threshold_v = torch.rand(
            int(math.ceil(span)),
            dimensions,
            dtype=torch.float,
            device=kwargs.get("device", None),
            generator=kwargs.get("generator", None),
        )

        hv = torch.empty(
            num_vectors,
            dimensions,
            # dtype=span_hv.dtype,
            # device=span_hv.device,
            dtype=dtype,
            device=device,
        ).as_subclass(vsa_tensor)

        # if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
        #     hv.block_size = span_hv.block_size

        for i in range(num_vectors):
            span_idx = int(i // levels_per_span)

            # special case: if we are on a span border (e.g. on the first or last levels)
            # then set the orthogonal vector directly.
            # This also prevents an index out of bounds error for the last level
            # when threshold_v[span_idx], and span_hv[span_idx + 1] are not available.
            if abs(i % levels_per_span) < 1e-12:
                hv[i] = span_hv[span_idx]
            else:
                level_within_span = i % levels_per_span
                # the threshold value from the start hv's perspective
                t = 1 - (level_within_span / levels_per_span)

                span_start_hv = span_hv[span_idx]
                span_end_hv = span_hv[span_idx + 1]
                hv[i] = torch.where(threshold_v[span_idx] < t, span_start_hv, span_end_hv)

        hv.requires_grad = requires_grad
        return hv
  


    # def random2(
    #     self,
    #     num_vectors: int,
    #     dimensions: int,
    #     *,
    #     generator=None,
    #     dtype=None,
    #     device=None,
    #     requires_grad=False,
    # ) -> "MAPTensor":
    #     r"""Creates a set of random independent hypervectors.

    #     """

    #     cls = self.__class__

    #     if dtype is None:
    #         dtype = torch.get_default_dtype()

    #     if dtype not in cls.supported_dtypes:
    #         name = cls.__name__
    #         options = ", ".join([str(x) for x in cls.supported_dtypes])
    #         raise ValueError(f"{name} vectors must be one of dtype {options}.")
        
    #     size = (num_vectors, dimensions)
    #     select = torch.empty(size, dtype=torch.bool, device=device)
    #     for i in range(0, num_vectors):
    #         index = self.tensorDict.popTensor()
    #         select[i] = self.groupIndex2Tensor(index, self.dform)
            
    #     return select











