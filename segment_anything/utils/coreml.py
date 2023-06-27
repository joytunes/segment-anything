# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from segment_anything.modeling import Sam
from .transforms import ResizeLongestSide

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op


@register_torch_op(override=True)
def repeat_interleave(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    repeats = inputs[1]

    # Assume the dimension to repeat along is 0. Adjust as necessary.
    dim = 0

    # Create equivalent of torch.arange(x.shape[dim] * repeats)
    arange_max = mb.mul(x=x.shape[dim], y=repeats)
    i = mb.range_1d(start=0, end=arange_max.val, step=1)

    # Equivalent of // operator
    i = mb.floor_div(x=i, y=repeats)

    # Equivalent of torch.index_select
    y = mb.gather(x=x, indices=i, axis=dim)

    # Add output to the context
    context.add(y, torch_name=node.name)

# This is supposed to be equivalent to the following pytorch code:
# # This method replaces torch.repeat_interleave with dim=0 as used here,
# # required as repeat_interleave is not supported by coremltools
# @staticmethod
# def _repeat_interleave(x, n, dim=0) -> torch.Tensor:
#     # e.g. x=[1, 2, 3], n=2 => returns [1, 1, 2, 2, 3, 3]
#     i = torch.as_tensor(torch.arange(x.size()[0] * n, device=x.device) // n, dtype=torch.int32)
#     y = torch.index_select(x, 0, i)
#     return y



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

        input_image_torch = image.contiguous()

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


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Tuple

from ..modeling import Sam
from .amg import calculate_stability_score


class SamPointDecoder(nn.Module):
    def __init__(
            self,
            not_model: Sam,
            return_single_mask: bool,
            use_stability_score: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = not_model.mask_decoder
        self.prompt_encoder = not_model.prompt_encoder
        self.mask_threshold = not_model.mask_threshold
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = (point_coords + 0.5) / self.prompt_encoder.input_image_size[0]
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self) -> torch.Tensor:
        return self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

    def select_masks(
            self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
            self,
            image_embeddings: torch.Tensor,
            points_and_box_coords: torch.Tensor,
            points_and_box_labels: torch.Tensor,
            ):
        sparse_embedding = self._embed_points(points_and_box_coords, points_and_box_labels)
        dense_embedding = self._embed_masks()

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, points_and_box_coords.shape[1])

        return scores, masks
