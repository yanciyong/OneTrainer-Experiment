from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class BaseModelSaver(metaclass=ABCMeta):

    @abstractmethod
    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # Import FSDP locally

        if isinstance(model, FSDP):
            model = model.module
        pass
