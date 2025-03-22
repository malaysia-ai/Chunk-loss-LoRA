import torch
import logging

try:
    from deepspeed.runtime.zero import partition_parameters
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from deepspeed.accelerator import get_accelerator

    def free_param(param) -> None:
        """Free underlying storage of a parameter."""
        if get_accelerator().on_accelerator(param.data):
            # need to make sure that we don't free the parameter while it is still
            # being used for computation
            if not get_accelerator().is_synchronized_device():
                param.data.record_stream(get_accelerator().current_stream())
        # param.data doesn't store anything meaningful in partitioned state
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    partition_parameters.free_param = free_param
except Exception as e:
    logging.warning('deepspeed failed to load, patching free_param failed')
    logging.warning(e)

from .ce import *
from .linear import *