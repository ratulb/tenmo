"""Tenmo GPU — Layer-0 GPU device abstractions.
   Must never import upward (tenmo.ndbuffer, tenmo.tensor, ...).
"""

from .device import (
    GPU,
    DeviceState,
    Device,
    CPU,
    DeviceType,
)
from .runtime import elementwise_launch_config
from .transfer import (
    host_to_device,
    device_to_host,
    device_to_host_strided,
    materialize_contiguous,
)
