# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Any, Dict, List, Tuple

from .transforms import ResizeLongestSide


class SamEmbedder(torch.nn.Module):
    def __init__(
            self,
            sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.not_model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    @torch.no_grad()
    def forward(self, image: torch.Tensor,
                ) -> torch.Tensor:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (torch.Tensor): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        """

        # Transform the image to the form expected by the model
        # input_image = self.transform.apply_image(image)
        # input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = image.contiguous() #[None, :, :, :]

        assert (
                len(input_image_torch.shape) == 4
                and input_image_torch.shape[1] == 3
                and max(*input_image_torch.shape[2:]) == self.not_model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.not_model.image_encoder.img_size}."

        input_image = self.not_model.preprocess(input_image_torch)
        image_embeddings = self.not_model.image_encoder(input_image)

        return image_embeddings

    @property
    def device(self) -> torch.device:
        return self.not_model.device
