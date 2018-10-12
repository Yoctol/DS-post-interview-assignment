from typing import Tuple, Union

import numpy as np

SupervisedData = Tuple[np.ndarray, np.ndarray]
UnsupervisedData = np.ndarray
Data = Union[SupervisedData, UnsupervisedData]
