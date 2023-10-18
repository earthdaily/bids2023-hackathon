from typing import List, Optional

from torch import Tensor, reshape, stack

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)


class PixelwiseLinear(Module):
    """
    PixelwiseLinear Class: Applies a sequence of 1x1 convolutions followed by PReLU activations.

    Parameters:
    - fin (List[int]): List of input feature dimensions for each layer.
    - fout (List[int]): List of output feature dimensions for each layer.
    - last_activation (Module, optional): Activation function to use for the last layer.

    Example:
    ```
    model = PixelwiseLinear([32, 16], [16, 8], last_activation=Sigmoid())
    ```
    """

    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)


class MixingBlock(Module):
    """
    MixingBlock Class: Combines two input feature maps (x and y) by interleaving their channels and
    applies a set of operations including depth-wise 2D convolution, PReLU activation, and instance normalization.

    Parameters:
    - ch_in (int): Number of input channels for each input tensor.
    - ch_out (int): Number of output channels for the resulting tensor.

    Example:
    ```
    mixing_layer = MixingBlock(32, 64)
    out = mixing_layer(x, y)
    ```

    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass for MixingBlock.
        Packing the tensors and interleaving the channels

        Parameters:
        - x (Tensor): First input tensor.
        - y (Tensor): Second input tensor.

        Returns:
        - Tensor: Output tensor after channel mixing and convolutional operations.
        """
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """
    MixingMaskAttentionBlock Class: Implements a sort of attention mechanism using grouped convolution
    to combine two input feature maps (x and y). Optionally applies an additional instance normalization
    at the end.

    Parameters:
    - ch_in (int): Number of input channels for each input tensor.
    - ch_out (int): Number of output channels for the mixed tensor.
    - fin (List[int]): List of input dimensions for the PixelwiseLinear layer.
    - fout (List[int]): List of output dimensions for the PixelwiseLinear layer.
    - generate_masked (bool): If True, applies an additional instance normalization at the end.

    Example:
    ```
    mixing_mask_layer = MixingMaskAttentionBlock(32, 64, [16, 32], [32, 16])
    out = mixing_mask_layer(x, y)
    ```
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(Module):
    """
    UpMask Class: Implements up-sampling followed by a sequence of convolutional layers.
    Optionally multiplies the up-sampled feature map with another input feature map (y).

    Parameters:
    - scale_factor (float): Factor by which the input tensor is up-sampled.
    - nin (int): Number of input channels.
    - nout (int): Number of output channels.

    Example:
    ```
    upmask_layer = UpMask(2, 32, 64)
    out = upmask_layer(x, y)
    ```
    """

    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)
