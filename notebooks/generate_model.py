import torch
import torchvision
import coremltools as ct
import sys
import os
sys.path.append('..')
from segment_anything import sam_model_registry
from segment_anything.utils.coreml import SamEmbedder

# checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
checkpoint = 'sam_vit_b_01ec64.pth'
model_type = 'vit_b'

checkpoint_dir = '.' #'/Users/anatoli/Documents/segment-anything/checkpoints'

class MyModelWrapper3(torch.nn.Module):
    def __init__(self, model):
        super(MyModelWrapper3, self).__init__()
        self.model = model

    def forward(self, *x):
        output = self.model(*x)
        # Modify output here
        return output


sam = sam_model_registry[model_type](checkpoint=os.path.join(checkpoint_dir, checkpoint))
mymodel3 = MyModelWrapper3(SamEmbedder(sam))
dummy_inputs_for_sam = {
    # The image as a torch tensor in 3xHxW format, already transformed for input to the model.
    'image': torch.randint(low=0, high=255, size=(1, 3, 1024, 1024), dtype=torch.float),
}

sam_traced_model = torch.jit.trace(mymodel3.eval(), tuple(dummy_inputs_for_sam.values()))


coreml_inputs = {
    'image': ct.ImageType(name='image', shape=dummy_inputs_for_sam['image'].shape, channel_first=True),
}
coreml_outputs = {
    "image_embeddings": ct.TensorType(name="image_embeddings")
}
embedder_model = ct.convert(
    sam_traced_model,
    outputs=list(coreml_outputs.values()),
    inputs=list(coreml_inputs.values()),
    minimum_deployment_target=ct.target.iOS15
)

embedder_model.save("sambedder.mlpackage")