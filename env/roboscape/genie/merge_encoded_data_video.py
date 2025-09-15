import numpy as np
import os
import json
from tqdm import *

v = "video"
dim = 16
split = "val"
root = "/tangyinzhou-tos-volc-engine/tyz/text_action_split_100000"
merged_root = os.path.join(root, "merged", split)
os.makedirs(merged_root, exist_ok=True)

# ---------- 第 1 轮：统计总帧数 ----------
total_images = 0
for part in trange(10, desc="Counting frames"):
    meta_path = os.path.join(root, f"part_{part}", split, "metadata.json")
    with open(meta_path, "r") as f:
        total_images += json.load(f)["num_images"]

# ---------- 创建最终大小的空 bin ----------
merged_bin = os.path.join(merged_root, f"{v}.bin")
merged_mmap = np.memmap(
    merged_bin, dtype=np.float32, mode="w+", shape=(total_images, dim, dim)
)
print("created mmap")
# ---------- 第 2 轮：分批拷入 ----------
offset = 0
batch_size = 10_000  # 可按磁盘性能调整
for part in trange(10, desc="Merging parts"):
    meta_path = os.path.join(root, f"part_{part}", split, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    part_mmap = np.memmap(
        os.path.join(root, f"part_{part}", split, f"{v}.bin"),
        dtype=np.float32,
        mode="r",
        shape=(meta["num_images"], dim, dim),
    )

    # 把当前 part 分批次拷到目标文件
    for start in tqdm(
        range(0, part_mmap.shape[0], batch_size),
        desc=f"Part {part}",
        leave=False,
    ):
        end = min(start + batch_size, part_mmap.shape[0])
        merged_mmap[offset + start : offset + end] = part_mmap[start:end]

    offset += part_mmap.shape[0]

merged_mmap.flush()
if not os.path.exists(
    f"/tangyinzhou-tos-volc-engine/tyz/text_action_split_100000/merged/{split}/metadata.json"
):
    with open(
        f"/tangyinzhou-tos-volc-engine/tyz/text_action_split_100000/merged/{split}/metadata.json",
        "w",
    ) as f:
        json.dump(
            {
                "token_dtype": "uint32",
                "s": 16,
                "h": 16,
                "w": 16,
                "vocab_size": 262144,
                "hz": 30,
                "tokenizer_ckpt": "data/magvit2.ckpt",
                "num_images": total_images,
            },
            f,
        )
