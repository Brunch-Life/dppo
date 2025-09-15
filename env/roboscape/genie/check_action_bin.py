import numpy as np


class SimlerWrapper:
    def __init__(self, num_bins_per_dim=255):
        self.num_bins_per_dim = num_bins_per_dim  # 每个维度的bin数量
        self.dim_bins = None  # [dim, num_bins+1]
        self.dim_bin_centers = None  # [dim, num_bins]
        self.bin_counts = None  # [dim, num_bins]，存储每个bin的样本数量

    def fit_bins(self, action_np):
        """
        重写：基于10%-90%区间划分bin，超出部分作为单独类别
        每个维度的bin结构：
        - bin0：所有 < q10 的样本
        - bin1 ~ binN：[q10, q90] 区间内的等距划分（N = num_bins_per_dim）
        - binN+1：所有 > q90 的样本
        """
        if len(action_np.shape) != 2:
            raise ValueError("输入action_np必须是(B, dim)形状的二维数组！")

        B, dim = action_np.shape
        self.dim_bins = []
        self.dim_bin_centers = []
        self.bin_counts = []
        self.sample_bin_indices = np.zeros((B, dim), dtype=int)

        for d in range(dim):
            dim_data = action_np[:, d]  # 第d维的所有样本

            # 步骤1：计算10%分位数（q10）和90%分位数（q90）
            q10 = np.percentile(dim_data, 10)  # 10%分位数（90%数据大于等于此值）
            q90 = np.percentile(dim_data, 90)  # 90%分位数（90%数据小于等于此值）
            # 处理q10 == q90的极端情况（如数据高度集中）
            if np.isclose(q10, q90):
                q10 = np.min(dim_data)
                q90 = np.max(dim_data)

            # 步骤2：生成bins边界
            # - 左侧边界：使用数据最小值（确保覆盖所有 < q10 的样本）
            # - 中间区间：[q10, q90] 内划分 num_bins_per_dim 个等距bin
            # - 右侧边界：使用数据最大值（确保覆盖所有 > q90 的样本）
            left_bound = np.min(dim_data)
            right_bound = np.max(dim_data)
            # 中间区间的bins（q10到q90）
            middle_bins = np.linspace(q10, q90, self.num_bins_per_dim + 1)
            # 完整bins：[left_bound] + middle_bins + [right_bound]
            # 注意：left_bound 可能 < q10，right_bound 可能 > q90
            bins = np.concatenate([[left_bound], middle_bins, [right_bound]])

            # 步骤3：计算每个bin的中心值
            bin_centers = []
            # bin0：< q10 的中心（left_bound 到 q10 的中点）
            bin0_center = (left_bound + q10) / 2.0
            bin_centers.append(bin0_center)
            # bin1 ~ binN：中间区间的bin中心
            middle_centers = (middle_bins[:-1] + middle_bins[1:]) / 2.0
            bin_centers.extend(middle_centers)
            # binN+1：> q90 的中心（q90 到 right_bound 的中点）
            binN1_center = (q90 + right_bound) / 2.0
            bin_centers.append(binN1_center)
            bin_centers = np.array(bin_centers)

            # 步骤4：统计每个bin的样本数量
            counts, _ = np.histogram(dim_data, bins=bins)

            # 步骤5：计算每个样本在该维度的bin索引
            sample_indices = np.digitize(dim_data, bins) - 1  # 转换为0开始的索引
            # 由于bins已包含min和max，索引不会超出范围，无需额外clip
            self.sample_bin_indices[:, d] = sample_indices

            # 保存当前维度的结果
            self.dim_bins.append(bins)
            self.dim_bin_centers.append(bin_centers)
            self.bin_counts.append(counts)

        # 转换为numpy数组方便索引
        self.dim_bins = np.array(self.dim_bins)
        self.dim_bin_centers = np.array(self.dim_bin_centers)
        self.bin_counts = np.array(self.bin_counts)  # 形状：[dim, num_bins_per_dim + 2]

    def get_bin_centers(self, action_np) -> np.ndarray:
        """获取输入动作对应的bin中心值（原有功能保持不变）"""
        if self.dim_bin_centers is None:
            raise ValueError("请先调用fit_bins方法生成bin中心！")

        B, dim = action_np.shape
        center_values = np.zeros((B, dim), dtype=np.float32)

        for d in range(dim):
            bins = self.dim_bins[d]
            bin_centers = self.dim_bin_centers[d]
            dim_data = action_np[:, d]
            indices = np.digitize(dim_data, bins) - 1
            indices = np.clip(indices, 0, self.num_bins_per_dim - 1)
            center_values[:, d] = bin_centers[indices]

        return center_values

    def show_bin_counts(self, dim=None, show_all=False):
        """
        展示每个bin的样本数量

        参数:
            dim: 可选，指定维度（如0,1,2），只展示该维度的bin数量
            show_all: 是否展示所有维度的bin数量（若dim不为None，此参数无效）
        """
        if self.bin_counts is None:
            raise ValueError("请先调用fit_bins方法统计bin样本数量！")

        total_dim = self.bin_counts.shape[0]  # 总维度数

        # 分维度展示（指定单个维度）
        if dim is not None:
            if dim < 0 or dim >= total_dim:
                raise ValueError(f"维度超出范围，有效维度为0~{total_dim-1}")

            print(f"\n===== 维度 {dim} 的bin样本数量 =====")
            print(
                f"维度 {dim} 共 {self.num_bins_per_dim} 个bin，总样本数：{self.bin_counts[dim].sum()}"
            )
            # 打印每个bin的索引和对应的样本数（可根据需要调整打印数量，避免过长）
            for bin_idx in range(self.num_bins_per_dim):
                print(f"bin {bin_idx:3d}: {self.bin_counts[dim][bin_idx]:5d} 个样本")
                # 若bin数量过多，可限制打印前N个和后N个
                # if bin_idx < 5 or bin_idx >= self.num_bins_per_dim -5:
                #     print(f"bin {bin_idx:3d}: {self.bin_counts[dim][bin_idx]:5d} 个样本")
                # elif bin_idx == 5:
                #     print("...")

        # 整体展示（所有维度）
        elif show_all:
            print(f"\n===== 所有维度的bin样本数量汇总 =====")
            print(f"总维度数：{total_dim}，每个维度的bin数量：{self.num_bins_per_dim}")
            for d in range(total_dim):
                total_samples = self.bin_counts[d].sum()
                # 统计非空bin的数量（可选）
                non_empty_bins = np.sum(self.bin_counts[d] > 0)
                print(
                    f"维度 {d}: 总样本数={total_samples:6d}，非空bin数量={non_empty_bins:3d}，"
                    f"平均每个bin样本数={total_samples / self.num_bins_per_dim:.2f}"
                )

        else:
            print("请指定维度（dim）或设置show_all=True以展示所有维度")


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据：10000个样本，5个维度

    action_np = np.memmap(
        "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta/merged/train/action.bin",
        dtype=np.float32,
        mode="r",
    ).reshape(
        -1, 7
    )  # 假设数据是(B, 7)形状（7个维度）
    B, dim = action_np.shape
    # 初始化并拟合bins
    wrapper = SimlerWrapper(num_bins_per_dim=5)  # 为了演示方便，每个维度只分10个bin
    wrapper.fit_bins(action_np)

    # 1. 展示单个维度的bin样本数量（如维度0）
    for d in range(dim):
        wrapper.show_bin_counts(dim=d)

    # 2. 展示所有维度的汇总信息
    wrapper.show_bin_counts(show_all=True)
