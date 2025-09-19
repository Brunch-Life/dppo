import numpy as np
import os
import json
from tqdm import *

root = "/manifold-obs/tangyinzhou/encoded_maniskill_delta_multi_view_small"
is_multiview = True
tasks = [x for x in os.listdir(root) if "2025" in x]
for split in ["train", "val"]:
    merged_root = os.path.join(root, "merged", split)
    os.makedirs(merged_root, exist_ok=True)
    # bins except for video
    # ---------- 第 1 轮：统计总帧数 ----------
    total_images = 0
    for task in tqdm(tasks, desc="Counting frames"):
        # meta_path = os.path.join(root, task, task, split, "metadata.json")
        meta_path = os.path.join(root, task, split, "metadata.json")
        with open(meta_path, "r") as f:
            total_images += json.load(f)["num_images"]

    # bins for videos
    merged_bin = os.path.join(merged_root, f"delta_action_10dim.bin")
    merged_mmap = np.memmap(
        merged_bin, dtype=np.float32, mode="w+", shape=(total_images, 20, 10)
    )
    print("created mmap")
    # ---------- 第 2 轮：分批拷入 ----------
    offset = 0
    batch_size = 10_000  # 可按磁盘性能调整
    for task in tqdm(tasks, desc="Merging parts"):
        # meta_path = os.path.join(root, task, task, split, "metadata.json")
        meta_path = os.path.join(root, task, split, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        part_mmap = np.memmap(
            # os.path.join(root, task, task, split, f"delta_action_10dim.bin"),
            os.path.join(root, task, split, f"delta_action_10dim.bin"),
            dtype=np.float32,
            mode="r",
            shape=(meta["num_images"], 20, 10),
        )

        # 把当前 part 分批次拷到目标文件
        for start in tqdm(
            range(0, part_mmap.shape[0], batch_size),
            desc=f"Task {task}",
            leave=False,
        ):
            end = min(start + batch_size, part_mmap.shape[0])
            merged_mmap[offset + start : offset + end] = part_mmap[start:end]

        offset += part_mmap.shape[0]

    merged_mmap.flush()
