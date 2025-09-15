import numpy as np
import os
import json
from tqdm import *

root = "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta"
tasks = [x for x in os.listdir(root) if "2025" in x]
for split in ["train", "val"]:
    merged_root = os.path.join(root, "merged", split)
    os.makedirs(merged_root, exist_ok=True)
    # bins except for video
    # ---------- 第 1 轮：统计总帧数 ----------
    total_images = 0
    for task in tqdm(tasks, desc="Counting frames"):
        meta_path = os.path.join(root, task, split, "metadata.json")
        with open(meta_path, "r") as f:
            total_images += json.load(f)["num_images"]

    # segment_ids # Need to adjust
    merged_bin = os.path.join(merged_root, f"segment_ids.bin")
    merged_mmap = np.memmap(
        merged_bin, dtype=np.int32, mode="w+", shape=(total_images, 1)
    )
    # ---------- 第 2 轮：分批拷入 ----------
    offset = 0
    id_offset = 0
    batch_size = 10000  # 可按磁盘性能调整
    for task in tqdm(tasks, desc=f"Merging tasks for segment_ids"):
        meta_path = os.path.join(root, task, split, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        part_mmap = np.memmap(
            os.path.join(root, task, split, f"segment_ids.bin"),
            dtype=np.int32,
            mode="r",
            shape=(meta["num_images"], 1),
        )

        # 把当前 part 分批次拷到目标文件
        for start in tqdm(
            range(0, part_mmap.shape[0], batch_size),
            desc=f"Task {task}",
            leave=False,
        ):
            end = min(start + batch_size, part_mmap.shape[0])
            segs = [x + id_offset for x in part_mmap[start:end]]
            merged_mmap[offset + start : offset + end] = segs

        offset += part_mmap.shape[0]
        id_offset += int(part_mmap[-1]) + 1
    merged_mmap.flush()

    for v, dim in zip(["action", "done", "text"], [7, 1, 768]):
        # ---------- 创建最终大小的空 bin ----------
        merged_bin = os.path.join(merged_root, f"{v}.bin")
        merged_mmap = np.memmap(
            merged_bin, dtype=np.float32, mode="w+", shape=(total_images, dim)
        )
        # ---------- 第 2 轮：分批拷入 ----------
        offset = 0
        batch_size = 1000  # 可按磁盘性能调整
        for task in tqdm(tasks, desc=f"Merging tasks for {v}"):
            meta_path = os.path.join(root, task, split, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            part_mmap = np.memmap(
                os.path.join(root, task, split, f"{v}.bin"),
                dtype=np.float32,
                mode="r",
                shape=(meta["num_images"], dim),
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

    # bins for videos
    merged_bin = os.path.join(merged_root, f"video.bin")
    merged_mmap = np.memmap(
        merged_bin, dtype=np.uint32, mode="w+", shape=(total_images, 16, 16)
    )
    print("created mmap")
    # ---------- 第 2 轮：分批拷入 ----------
    offset = 0
    batch_size = 10_000  # 可按磁盘性能调整
    for task in tqdm(tasks, desc="Merging parts"):
        meta_path = os.path.join(root, task, split, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        part_mmap = np.memmap(
            os.path.join(root, task, split, f"video.bin"),
            dtype=np.uint32,
            mode="r",
            shape=(meta["num_images"], 16, 16),
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

    # text
    # 在整个循环外只打开一次 f_out
    with open(os.path.join(merged_root, "text.txt"), "w") as f_out:  # 第一次清空文件
        for task in tasks:
            file_path = os.path.join(root, task, split, "text.txt")
            if os.path.exists(file_path):
                with open(file_path) as f_in:
                    f_out.write(f_in.read())
    # with open(os.path.join(merged_root, f"text.txt"), "a") as f_out, open(
    #     os.path.join(root, task, split, f"text.txt")
    # ) as f_in:
    #     f_out.write(f_in.read())

    # metadata.json
    with open(
        f"{merged_root}/metadata.json",
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
