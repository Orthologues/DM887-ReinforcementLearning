import numpy as np
from torch import DeviceObjType, Tensor, nn, tensor, cat as concat_tensors
from typing import Tuple, List, Union, Any, Dict

import torch

"""
The class to construct the 
"""
class MeanStdRunner:
    def __init__(self, shape: Tuple[int], eps=1e-4) -> None:
        self.mean = np.zeros(shape, 'float64') 
        self.variance = np.ones(shape, 'float64')
        self.count = eps 


    def update(self, x: np.ndarray) -> None:
        batch_mean: np.ndarray = np.mean(x, axis=0)
        batch_variance: np.ndarray = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_variance, batch_count)


    def update_from_moments(self, batch_mean: np.float64, batch_variance: np.float64, batch_count: float) -> None:
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.variance * self.count
        m_b = batch_variance * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.variance = new_var
        self.count = new_count
    

"""
The subsequent classes are used for defining the normalizer classes with a base version as a root parent class
"""
class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def __call__(self, x):
        raise NotImplementedError("This method should be overridden.")

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return
    

class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, clip=10.0, epsilon=1e-8):
        super().__init__(self)
        self.rms = None
        self.clip = clip
        self.eps = epsilon

    def __call__(self, x: Union[List[Any], Tuple[Any]]) -> None:
        x: np.ndarray = np.asarray(x)
        if self.rms is None:
            self.rms = MeanStdRunner(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.variance + self.eps),
                       -self.clip, self.clip)
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        return { 'mean': self.rms.mean, 'variance': self.rms.variance }
    

    @property
    def mean_std_getter(self) -> MeanStdRunner:
        return self.rms


    @mean_std_getter.setter
    def load_state_dict(self, data) -> None:
        self.rms.mean = data['mean']
        self.rms.variance = data['variance']


class RescalingNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        super().__init__(self)
        self.coef = coef

    def __call__(self, x):
        return self.coef * x
    


"""
How to use it:
1. Initialize the instnace self.normalizer = AtariImageNormalizer()
2. __call__ method: normalized_state = self.normalizer(stack_of_processed_states, new_atari_state)
"""
class AtariImageNormalizer(RescalingNormalizer):
    def __init__(self, device: DeviceObjType, width, height, frame_stack_size, coef = 1.0 / 255, rbg2greyscale=True):
        super().__init__(coef)
        self.frame_stack_size = frame_stack_size
        self.width = width
        self.height = height
        self.rbg2greyscale = rbg2greyscale
        self.device = device
    

    """
    @param stack_of_frames: torch.Tensor 
        desired stack_of_frames.shape = (4, 84, 84)
    @param state: numpy.ndarray
        desired state.shape = (210, 160, 3)
    """
    def __call__(self, stack_of_frames: Tensor, new_frame: np.ndarray) -> Tensor:
        
        if not tuple(stack_of_frames.shape) == (self.frame_stack_size, self.width, self.height):
            raise NotImplementedError(f"The input stack of frames must be of size (4, 84, 84)")

        if not tuple(new_frame.shape) == (210, 160, 3):
            raise NotImplementedError("The Atari state frame must have a shape of (210, 160, 3)!")

        # $new_frame has a shape as (210, 160, 3), therefore, reshaping is necessary
        x = tensor(new_frame).to(device=self.device, dtype=torch.float32).detach().permute(2, 0, 1)
        x = super().__call__(x)
        if self.rbg2greyscale:
            # Convert RGB to grayscale
            x = 0.2989 * x[0, :, :] + 0.5870 * x[1, :, :] + 0.1140 * x[2, :, :]
            # Add two dimensions to the start, the input frame has a shape at (1, 1, 210, 160) afterwards 
            x = x.unsqueeze(0)  
            x = x.unsqueeze(0)
        # input param "x" of nn.functional.interpolate must be of shape (batch_size, n_channels, width, height), it would have a shape at (1, 1, 84, 84) afterwards by default
        x: Tensor = nn.functional.interpolate(x, size=(self.width, self.height), mode='bilinear', align_corners=False)
        # output tensor will be of shape (batch_size, width, height)
        return concat_tensors((stack_of_frames[1:, :, :], x.squeeze(0)), dim=0)



"""
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
"""
class SignNormalizer(BaseNormalizer):
    def __init__(self):
        super().__init__(self)

    def __call__(self, x):
        return np.sign(x)

    