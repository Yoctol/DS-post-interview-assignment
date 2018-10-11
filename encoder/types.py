from typing import Dict, Tuple, Union

import numpy as np

from .task import SupervisedTask, UnsupervisedTask

SupervisedData = Tuple[np.ndarray, np.ndarray]
UnsupervisedData = np.ndarray
Data = Union(SupervisedData, UnsupervisedData)

MultiSupervisedData = Dict[SupervisedTask, SupervisedData]
MultiUnsupervisedData = Dict[UnsupervisedTask, UnsupervisedData]
MultiTaskData = Union[MultiSupervisedData, MultiUnsupervisedData]
