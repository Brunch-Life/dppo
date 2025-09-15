import numpy as np
import os
import json
from tqdm import *

root = "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta"
tasks = [x for x in os.listdir(root) if "2025" in x]
for split in ["train", "val"]:
    merged_roots = [
        os.path.join(root, "merged_pick", split),
        os.path.join(root, "merged_place", split),
    ]
    task_num = len(merged_roots)
    for x in merged_roots:
        os.makedirs(x, exist_ok=True)
    # bins except for video
    # ---------- 第 1 轮：统计总帧数 ----------
    total_images = [0 for _ in range(task_num)]
    seg2taskid = {x: [] for x in os.listdir(root) if "2025" in x}
    for task in tqdm(tasks, desc=f"Merging tasks for segment_ids"):
        meta_path = os.path.join(root, task, split, "text.txt")
        with open(meta_path, "r") as f:
            meta = f.readlines()
        for text in meta:
            task_id = int("Place" in text)
            seg2taskid[task].append(task_id)
            total_images[task_id] += 1
    # segment_ids # Need to adjust
    merged_mmap = []
    for idx, merged_root in enumerate(merged_roots):
        merged_bin = os.path.join(merged_root, f"segment_ids.bin")
        merged_mmap.append(
            np.memmap(
                merged_bin, dtype=np.int32, mode="w+", shape=(total_images[idx], 1)
            )
        )
    # ---------- 第 2 轮：分批拷入 ----------
    offset = [0 for _ in total_images]
    id_offset = [0 for _ in total_images]
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
        starts = [0 for _ in range(task_num)]
        ends = [0 for _ in range(task_num)]
        # 把当前 part 分批次拷到目标文件
        for start in tqdm(
            range(0, part_mmap.shape[0], batch_size),
            desc=f"Task {task}",
            leave=False,
        ):
            end = min(start + batch_size, part_mmap.shape[0])
            segs = [[] for _ in range(task_num)]
            for i, x in enumerate(part_mmap[start:end]):
                task_id = seg2taskid[task][i + start]
                segs[task_id].append(x // task_num + id_offset[task_id])
            for task_id in range(task_num):
                ends[task_id] = ends[task_id] + len(segs[task_id])
                if segs[task_id]:
                    merged_mmap[task_id][
                        offset[task_id]
                        + starts[task_id] : offset[task_id]
                        + ends[task_id]
                    ] = segs[task_id]
                starts[task_id] = ends[task_id]
        for task_id in range(task_num):
            offset[task_id] += ends[task_id]
            id_offset[task_id] = (
                int(segs[task_id][-1]) + 1
                if segs[task_id]
                else int(segs[1 - task_id][-1]) + 1
            )
    for task_id in range(task_num):
        merged_mmap[task_id].flush()

    for v, dim in zip(["action", "done", "text"], [7, 1, 768]):
        # ---------- 创建最终大小的空 bin ----------
        for task_id, merged_root in enumerate(merged_roots):
            merged_bin = os.path.join(merged_root, f"{v}.bin")
            merged_mmap[task_id] = np.memmap(
                merged_bin,
                dtype=np.float32,
                mode="w+",
                shape=(total_images[task_id], dim),
            )
        # ---------- 第 2 轮：分批拷入 ----------
        offset = [0 for _ in total_images]
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
            starts = [0 for _ in range(task_num)]
            ends = [0 for _ in range(task_num)]
            # 把当前 part 分批次拷到目标文件
            for start in tqdm(
                range(0, part_mmap.shape[0], batch_size),
                desc=f"Task {task}",
                leave=False,
            ):
                end = min(start + batch_size, part_mmap.shape[0])
                data = [[] for _ in range(task_num)]
                for i, x in enumerate(part_mmap[start:end]):
                    task_id = seg2taskid[task][i + start]
                    data[task_id].append(x)
                for task_id in range(task_num):
                    ends[task_id] = ends[task_id] + len(data[task_id])
                    if data[task_id]:
                        merged_mmap[task_id][
                            offset[task_id]
                            + starts[task_id] : offset[task_id]
                            + ends[task_id]
                        ] = data[task_id]
                    starts[task_id] = ends[task_id]
                # merged_mmap[offset + start : offset + end] = part_mmap[start:end]
            for task_id in range(task_num):
                offset[task_id] += ends[task_id]
            # offset += part_mmap.shape[0]
        for task_id in range(task_num):
            merged_mmap[task_id].flush()

    # bins for videos
    for task_id, merged_root in enumerate(merged_roots):
        merged_bin = os.path.join(merged_root, f"video.bin")
        merged_mmap[task_id] = np.memmap(
            merged_bin,
            dtype=np.uint32,
            mode="w+",
            shape=(total_images[task_id], 16, 16),
        )
    print("created mmap")
    # ---------- 第 2 轮：分批拷入 ----------
    offset = [0 for _ in range(task_num)]
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
        starts = [0 for _ in range(task_num)]
        ends = [0 for _ in range(task_num)]
        # 把当前 part 分批次拷到目标文件
        for start in tqdm(
            range(0, part_mmap.shape[0], batch_size),
            desc=f"Task {task}",
            leave=False,
        ):
            end = min(start + batch_size, part_mmap.shape[0])
            data = [[] for _ in range(task_num)]
            for i, x in enumerate(part_mmap[start:end]):
                task_id = seg2taskid[task][i + start]
                data[task_id].append(x)
            for task_id in range(task_num):
                ends[task_id] = ends[task_id] + len(data[task_id])
                if data[task_id]:
                    merged_mmap[task_id][
                        offset[task_id]
                        + starts[task_id] : offset[task_id]
                        + ends[task_id]
                    ] = data[task_id]
                starts[task_id] = ends[task_id]
            # merged_mmap[offset + start : offset + end] = part_mmap[start:end]
        for task_id in range(task_num):
            offset[task_id] += ends[task_id]
        # offset += part_mmap.shape[0]

    for task_id in range(task_num):
        merged_mmap[task_id].flush()

    # text
    # 在整个循环外只打开一次 f_out
    keywords = ["Pick", "Place"]
    for task_id, merged_root in enumerate(merged_roots):
        with open(
            os.path.join(merged_roots[task_id], "text.txt"), "w"
        ) as f_out:  # 第一次清空文件
            for task in tasks:
                file_path = os.path.join(root, task, split, "text.txt")
                if os.path.exists(file_path):
                    with open(file_path) as f_in:
                        for line in f_in:  # 逐行读取
                            if keywords[task_id] in line:  # 自定义判断函数
                                f_out.write(line)

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
                    "num_images": total_images[task_id],
                },
                f,
            )
