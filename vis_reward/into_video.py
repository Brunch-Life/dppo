import cv2
import os


def images_to_video():
    # 获取当前目录
    current_dir = os.getcwd()
    # 获取当前目录下所有图片文件
    image_files = [
        f
        for f in os.listdir(current_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # 按图片名称中的数字排序
    def get_image_number(filename):
        try:
            # 提取文件名中的数字部分
            number = int(os.path.splitext(filename)[0])
            return number
        except ValueError:
            return float("inf")  # 如果文件名不是数字，放到最后

    image_files.sort(key=get_image_number)

    if not image_files:
        print("当前目录下没有找到图片文件。")
        return

    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(os.path.join(current_dir, image_files[0]))
    height, width, _ = first_image.shape

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(current_dir, "output_video.mp4")
    # 将帧率从 1.0 修改为 5.0
    video = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))

    # 遍历所有图片并写入视频
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(current_dir, image_file)
        frame = cv2.imread(image_path)
        # 在左上角添加当前图片序号
        text = f"Step {idx}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video.write(frame)

    # 释放资源
    video.release()
    print(f"视频已生成，路径为: {video_path}")


if __name__ == "__main__":
    images_to_video()
