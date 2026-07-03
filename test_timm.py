import torch
import timm

url = "https://huggingface.co/david-shavin/SnD/resolve/main/dinov2_small_snd.pth"
state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=False,
    num_classes=0,
    dynamic_img_size=True,
    dynamic_img_pad=False,
)
model.load_state_dict(state_dict, strict=False)
model.eval()

x = torch.randn(1, 3, 224, 224)
out = model.forward_features(x)
print(type(out))
if isinstance(out, dict):
    print(out.keys())
elif isinstance(out, torch.Tensor):
    print(out.shape)
else:
    print(out)
