from env.roboscape.genie.data import RawTokenDataset
from env.roboscape.genie.eval_utils import compute_lpips, decode_tokens
from env.roboscape.genie.visualize import decode_latents_wrapper
import imageio
from env.roboscape.genie.genie.st_mask_git import GenieConfig
import os
import numpy as np
import cv2
from torchvision import transforms
import torch

config = GenieConfig.from_pretrained(
    "/ML-vePFS/tangyinzhou/RoboScape-R/dppo/env/roboscape/genie/genie/configs/magvit_n32_h8_d512_action_done.json"
)
dataset = RawTokenDataset(
    "/manifold-obs/tangyinzhou/encoded_maniskill_delta_multi_view_wooden/20250907_210135/20250907_210135",
    config=config,
    window_size=16,
    stride=4,
    split="train",
    use_action=True,
    use_text=False,
    hybrid_sample=False,
    use_target_action=True,
)
example_THW = (
    dataset[0]["input_ids"].reshape(16, -1, 16)[15][:16, :].reshape(1, 1, 16, 16)
)
wm_obs = decode_tokens(example_THW.cpu(), decode_latents_wrapper())
image = wm_obs.reshape(3, 256, 256).permute(1, 2, 0).cpu().numpy()
filename = f"/ML-vePFS/tangyinzhou/RoboScape-R/wm_obs.png"
imageio.imwrite(filename, image)

video_path = os.listdir(
    "/manifold-obs/bingwen/Datasets/wooden/plate/TabletopPickPlaceEnv-v1/20250907_210135/nonstop_plate_wooden/success"
)[0]
data = np.load(
    f"/manifold-obs/bingwen/Datasets/wooden/plate/TabletopPickPlaceEnv-v1/20250907_210135/nonstop_plate_wooden/success/{video_path}",
    allow_pickle=True,
)["arr_0"].tolist()
resized_frames = []
resize_transform = transforms.Resize((256, 256))
for jpeg_bytes in data["observation"]["rgb"]:
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC-RGB uint8
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CHW 0-1
    tensor = resize_transform(tensor)  # ← 你定义的 transforms.Resize(256,256)
    resized_frames.append(tensor)
data_obs = resized_frames[15 * 4]
image = (data_obs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
filename = f"/ML-vePFS/tangyinzhou/RoboScape-R/data_obs.png"
imageio.imwrite(filename, image)
