from typing import Dict, Tuple

import numpy as np

from .task import SupervisedTask, UnsupervisedTask

SupervisedData = Tuple[np.ndarray, np.ndarray]
MultiSupervisedData = Dict[SupervisedTask, SupervisedData]
MultiUnsupervisedData = Dict[UnsupervisedTask, np.ndarray]
