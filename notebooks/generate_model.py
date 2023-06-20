import torch
import coremltools as ct
import sys
import os
sys.path.append('..')
from segment_anything import sam_model_registry
from segment_anything.utils.coreml import SamEmbedder, SamPointDecoder

checkpoint_dir = '.' #'/Users/anatoli/Documents/segment-anything/checkpoints'

model_type = 'vit_b' # 'vit_b', 'vit_l' or 'vit_h'
if model_type == 'vit_h':
    checkpoint = 'sam_vit_h_4b8939.pth'
elif model_type == 'vit_b':
    checkpoint = 'sam_vit_b_01ec64.pth'
elif model_type == 'vit_l':
    checkpoint = 'sam_vit_l_0b3195.pth'
else:
    raise ValueError(f'Unknown model type {model_type}, model type must be one of "vit_b", "vit_l", "vit_h"')


sam = sam_model_registry[model_type](checkpoint=os.path.join(checkpoint_dir, checkpoint))
point_decoder_model = SamPointDecoder(sam, return_single_mask=False)
image_embedder_model = SamEmbedder(sam)

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
point_decoder_dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
}
image_embedder_dummy_inputs = {
    # The image as a torch tensor in 3xHxW format, already transformed for input to the model.
    'image': torch.randint(low=0, high=255, size=(1, 3, 1024, 1024), dtype=torch.float),
}


point_decoder_trace = torch.jit.trace(point_decoder_model.eval(), tuple(point_decoder_dummy_inputs.values()))
image_embedder_trace = torch.jit.trace(image_embedder_model.eval(), tuple(image_embedder_dummy_inputs.values()))

# Save the traced model
torch.jit.save(point_decoder_trace, f'point_decoder_{model_type}_model.pt')
torch.jit.save(image_embedder_trace, f'image_embedder_{model_type}_model.pt')

point_decoder_coreml_inputs = {
    "image_embeddings": ct.TensorType(name="image_embeddings", shape=point_decoder_dummy_inputs["image_embeddings"].size()),
    "point_coords": ct.TensorType(name="point_coords", shape=point_decoder_dummy_inputs["point_coords"].size()),
    "point_labels": ct.TensorType(name="point_labels", shape=point_decoder_dummy_inputs["point_labels"].size()),
    "mask_input": ct.TensorType(name="mask_input", shape=point_decoder_dummy_inputs["mask_input"].size()),
    "has_mask_input": ct.TensorType(name="has_mask_input", shape=point_decoder_dummy_inputs["has_mask_input"].size()),
}
point_decoder_coreml_outputs = {
    "iou_predictions": ct.TensorType(name="iou_predictions"),
    "low_res_masks": ct.TensorType(name="low_res_masks")
}
point_decoder_coreml_model = ct.convert(
    point_decoder_trace,
    outputs=list(point_decoder_coreml_outputs.values()),
    inputs=list(point_decoder_coreml_inputs.values()),
    minimum_deployment_target=ct.target.iOS15
)

image_embedder_coreml_inputs = {
    'image': ct.ImageType(name='image', shape=image_embedder_dummy_inputs['image'].shape, channel_first=True),
}
image_embedder_coreml_outputs = {
    "image_embeddings": ct.TensorType(name="image_embeddings")
}
image_embedder_coreml_model = ct.convert(
    image_embedder_trace,
    outputs=list(image_embedder_coreml_outputs.values()),
    inputs=list(image_embedder_coreml_inputs.values()),
    minimum_deployment_target=ct.target.iOS15
)

point_decoder_coreml_model.save(f"point_decoder_{model_type}.mlpackage")
image_embedder_coreml_model.save(f"image_embedder_{model_type}.mlpackage")