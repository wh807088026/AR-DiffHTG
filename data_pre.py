# 文件名: preprocess_images.py (已添加过滤和动态命名功能)

import os
from PIL import Image
from tqdm import tqdm
import argparse
import re
import math

# [新增] 将预定义的字符集作为全局常量
ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
# [新增] 转换为集合(Set)，以获得极快的查找速度
ALPHABET_SET = set(ALPHABET)


def parse_filename(filename: str) -> (str, str):
    """
    智能解析复杂的文件名。
    采用固定位置分割：前4个连字符分割的部分为新文件名，其余为标签。
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('-')

    if len(parts) > 4:
        new_basename = '-'.join(parts[:4])
        label = '-'.join(parts[4:])
        return new_basename, label
    else:
        return basename, ""


# [核心改动 1] 在函数签名中添加 strict_alphabet 参数
def preprocess_images(source_dir, dest_dir, target_height=64, min_width=32, mode='padding',
                      gt_filename="ground_truth.txt", strict_alphabet=False):
    """
    遍历、处理图像，并生成一个包含文件名和标签对照的 txt 文件。
    """
    image_paths = []
    ground_truth_lines = []
    skipped_count = 0  # [新增] 用于计数被过滤掉的样本

    print(f"正在扫描源文件夹: {source_dir}...")
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("在源文件夹中没有找到任何支持的图像文件。")
        return

    print(f"共找到 {len(image_paths)} 张图像，开始处理... 模式: {mode.upper()}")
    if strict_alphabet:
        print("INFO: 已开启严格字符集过滤模式。")

    for src_path in tqdm(image_paths, desc="正在处理图像"):
        try:
            original_filename = os.path.basename(src_path)
            new_basename, label = parse_filename(original_filename)

            if not label:
                print(f"\n警告：无法从文件名 {original_filename} 中解析出标签，已跳过。")
                continue

            # --- [核心改动 2: 重新加入过滤逻辑] ---
            if strict_alphabet:
                # 检查标签中的每个字符是否都在预定义的字符集内
                is_valid = all(char in ALPHABET_SET for char in label)
                if not is_valid:
                    skipped_count += 1
                    continue  # 如果包含非法字符，则跳过此文件的后续所有处理
            # --- [改动结束] ---

            author_id = os.path.basename(os.path.dirname(src_path))
            gt_line = f"{author_id},{new_basename} {label}"
            ground_truth_lines.append(gt_line)

            new_filename_with_ext = new_basename + '.png'
            relative_dir = os.path.relpath(os.path.dirname(src_path), source_dir)
            dest_folder = os.path.join(dest_dir, relative_dir)
            dest_path = os.path.join(dest_folder, new_filename_with_ext)
            os.makedirs(dest_folder, exist_ok=True)

            with Image.open(src_path) as img:
                # ... (图像处理逻辑与上一版完全相同，这里省略以保持简洁) ...
                img_gray = img.convert("L")
                original_width, original_height = img_gray.size
                if original_height > 0:
                    final_img_to_save = None
                    aspect_ratio = original_width / original_height
                    calculated_width = int(target_height * aspect_ratio)
                    base_width = max(calculated_width, min_width)
                    final_width = math.ceil(base_width / 32) * 32
                    if mode == 'padding':
                        resized_img = img_gray.resize((calculated_width, target_height), Image.Resampling.LANCZOS)
                        padded_img = Image.new("L", (final_width, target_height), 255)
                        paste_x = (final_width - calculated_width) // 2
                        padded_img.paste(resized_img, (paste_x, 0))
                        final_img_to_save = padded_img
                    elif mode == 'stretching':
                        resized_img = img_gray.resize((final_width, target_height), Image.Resampling.LANCZOS)
                        final_img_to_save = resized_img
                    if final_img_to_save:
                        final_img_to_save.save(dest_path)

        except Exception as e:
            print(f"\n处理文件 {src_path} 时出错: {e}")

    # [新增] 在处理结束后报告过滤掉的样本数量
    if strict_alphabet:
        print(f"INFO: 严格字符集过滤完成，共跳过 {skipped_count} 个样本。")

    if ground_truth_lines:
        parent_dir = os.path.dirname(os.path.normpath(dest_dir))
        gt_filepath = os.path.join(parent_dir, gt_filename)
        if not parent_dir: gt_filepath = gt_filename

        print(f"\n正在写入标注文件到: {gt_filepath}")
        try:
            # 确保父目录存在
            if parent_dir: os.makedirs(parent_dir, exist_ok=True)
            with open(gt_filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ground_truth_lines))
            print("标注文件写入成功！")
        except Exception as e:
            print(f"写入标注文件时出错: {e}")

    print(f"\n处理完成！所有图像已重命名并保存到: {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理图像数据集，固定高度，重命名文件，并生成标注。")
    parser.add_argument("--source", type=str,
                        default='/mnt/ssd4T/FID_KID_HWD_GS/CVL_REAL/cvl-database-1-1/testset/words', help="原始图像所在的源文件夹路径。")
    parser.add_argument("--dest", type=str,
                        default='/mnt/ssd4T/FID_KID_HWD_GS/CVL64-new/train', help="保存处理后图像的目标文件夹路径。")
    parser.add_argument("--height", type=int, default=64, help="处理后图像的固定高度。")
    parser.add_argument("--min_width", type=int, default=32, help="处理后图像的最小宽度。")
    parser.add_argument("--mode", type=str, default="padding", choices=['padding', 'stretching'],
                        help="宽度不足时的处理模式。")

    # [新增] 添加严格字符集过滤的选项
    parser.add_argument(
        "--strict_alphabet",
        action='store_true',
        default=True,
        help="如果使用此选项，将过滤掉标签不完全属于预定义字符集的图像。"
    )

    args = parser.parse_args()

    # --- [核心改动 2: 动态生成标注文件名] ---
    # 移除 --gt_file 参数，改为自动生成
    dest_path_norm = os.path.normpath(args.dest)
    last_folder = os.path.basename(dest_path_norm)
    parent_folder = os.path.basename(os.path.dirname(dest_path_norm))

    # 根据您的规则 'CVL64-new/test' -> 'CVL64_test.txt'
    base_gt_name = parent_folder.replace('-new', '')
    gt_filename = f"{base_gt_name}_{last_folder}.txt"
    print(f"INFO: 自动生成的标注文件名为: {gt_filename}")
    # --- [改动结束] ---

    preprocess_images(args.source, args.dest, args.height, args.min_width, args.mode, gt_filename, args.strict_alphabet)