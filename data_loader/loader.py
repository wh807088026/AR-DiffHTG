import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F

text_path_IAM = {'train': 'data/IAM64_train.txt',
                 'test': 'data/IAM64_test.txt'}
text_path_CVL = {'train': 'data/CVL64_train.txt',
                 'test': 'data/CVL64_test.txt'}

generate_type_IAM = {'iv_s': ['train', 'data/in_words.txt'],
                 'iv_u': ['test', 'data/in_words.txt'],
                 'oov_s': ['train', 'data/oov_words.txt'],
                 'oov_u': ['test', 'data/oov_words.txt'],
                 'test_fake': ['test', 'data/test_words'],
                 'fid_test': ['test', 'data/IAM64_test.txt'],
                 'fid_train': ['train', 'data/IAM64_train.txt']}

generate_type_CVL = {'iv_s': ['train', 'data/in_words.txt'],
                 'iv_u': ['test', 'data/in_words.txt'],
                 'oov_s': ['train', 'data/oov_words.txt'],
                 'oov_u': ['test', 'data/oov_words.txt'],
                 'test_fake': ['test', 'data/test_words'],
                 'fid_test': ['test', 'data/CVL64_test.txt'],
                 'fid_train': ['train', 'data/CVL64_train.txt']}

# define the letters and the width of style image
letters = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
style_len = 352

"""prepare the IAM dataset for training"""


class ContentData:
    """
    一个辅助类，用于将文本标签转换为字符图像张量。
    （逻辑来自您提供的原始 loader.py）
    """

    def __init__(self, content_type='unifont'):
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self._get_symbols(content_type)

    def _get_symbols(self, input_type):
        # !! 注意 !!: 请确保此路径指向您正确的 unifont.pickle 文件
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            # 确保即使 pickle 文件中缺少某个字符，也能继续运行
            if ord(char) in symbols:
                symbol = torch.from_numpy(symbols[ord(char)]).float()
                contents.append(symbol)
            else:
                # 如果缺少字符，用一个零矩阵代替
                print(f"Warning: Character '{char}' not found in {input_type}.pickle. Using a blank image.")
                contents.append(torch.zeros(16, 16))  # 假设字符图像尺寸为 16x16

        contents.append(torch.zeros_like(contents[0]))  # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents

    def get_content(self, label_str):
        word_indices = [self.letter2index[char] for char in label_str if char in self.letter2index]
        content_ref = self.con_symbols[word_indices]
        # 反转颜色 (白色背景，黑色字体)
        content_ref = 1.0 - content_ref
        # 返回 [N, 16, 16] 的张量，其中 N 是标签中的字符数
        return content_ref


class FidGenerationDataset(Dataset):
    """
    [新] 用于 FID 批量生成的统一数据集（稳健版）。
    __getitem__ 只返回原始数据，所有处理都交给 collate_fn。
    """

    def __init__(self, image_path, text_path, content_loader, transform=None):
        super().__init__()
        self.image_path = image_path
        # content_loader 和 transform 都是在 collate_fn 中使用，但在这里传入
        self.content_loader = content_loader
        self.transform = transform
        self.samples = self._load_samples(text_path)

    def _load_samples(self, data_path):
        """在初始化时，一次性解析好语料文件，创建样本列表。"""
        print("INFO: Parsing corpus file and matching samples...")
        samples = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                parts = line.split(' ')
                image_info, transcription = parts[0], parts[1]
                s_id, image_name_no_ext = image_info.split(',')
                image_name = image_name_no_ext + '.png'
                style_path = os.path.join(self.image_path, s_id, image_name)
                if os.path.exists(style_path):
                    samples.append({
                        "style_path": style_path,
                        "content_label": transcription,
                        "output_name": image_name,
                        "wid": s_id
                    })
        print(f"INFO: Found {len(samples)} valid samples for generation.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 只加载 PIL Image 和字符串，不进行任何 tensor 转换
        style_image_pil = Image.open(sample["style_path"]).convert("L")  # 使用灰度图以匹配原始逻辑

        return {
            "style_pil": style_image_pil,
            "content_label": sample["content_label"],
            "output_name": sample["output_name"],
            "wid": sample["wid"],
        }


def fid_collate_fn(batch, content_loader, transform):
    """
    [修正版] 为 FidGenerationDataset 准备的 collate_fn。
    修正了处理4D内容张量时的维度错误。
    """
    # --- 风格图像处理部分 (保持不变) ---
    style_images_pil = [item['style_pil'] for item in batch]
    style_tensors = [transform(img) for img in style_images_pil]
    height = style_tensors[0].shape[1]
    max_style_w = max(img.shape[2] for img in style_tensors)
    padded_styles = torch.ones([len(batch), 1, height, max_style_w], dtype=torch.float32) * -1
    for i, img in enumerate(style_tensors):
        padded_styles[i, :, :, :img.shape[2]] = img
    padded_styles_pair = torch.stack([padded_styles, padded_styles], dim=1).squeeze(2)

    # --- 内容序列处理部分 (核心修正) ---
    content_labels = [item['content_label'] for item in batch]
    # content_images_4d 是一个 [1, N_i, 16, 16] 形状的 4D 张量列表
    content_images_4d = [content_loader.get_content(label) for label in content_labels]

    # 1. [修正] 从每个 4D 张量的第二维 (shape[1]) 获取正确的序列长度 N_i
    max_content_len = max(seq.shape[1] for seq in content_images_4d)

    # 2. 创建一个正确大小的填充容器
    padded_contents = torch.zeros([len(batch), max_content_len, 16, 16], dtype=torch.float32)

    # 3. 循环并正确赋值
    for i, seq_4d in enumerate(content_images_4d):
        # [修正] 在赋值前，将 4D 张量降维成 3D 的 [N_i, 16, 16]
        seq_3d = seq_4d.squeeze(0)
        current_len = seq_3d.shape[0]
        # 现在是将 3D 的 seq_3d 赋给 3D 的切片，维度完全匹配
        padded_contents[i, :current_len, :, :] = seq_3d

    # --- 其他元数据 (保持不变) ---
    output_names = [item['output_name'] for item in batch]
    wids = [item['wid'] for item in batch]
    content_lengths = [len(label) for label in content_labels]

    return {
        "style": padded_styles_pair,
        "content": padded_contents,
        "output_name": output_names,
        "wid": wids,
        "content_lengths": content_lengths
    }


class IAMDataset(Dataset):
    def __init__(self, image_path, style_path, dataset, type, content_type='unifont', max_len=9):
        self.max_len = max_len
        self.style_len = style_len
        text_path = text_path_IAM
        if dataset == 'CVL':
            text_path = text_path_CVL
        self.data_dict = self.load_data(text_path[type])
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)

        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label': transcription}
                idx += 1
        return full_dict

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2)  # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]

        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])

        '''style images'''
        style_images = [style_image / 255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        return new_style_images

    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0]))  # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents

    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32)  # [2, h , w] achor and positive

        return {'img': image,
                'content': label,
                'style': style_ref,
                'wid': int(wr_id),
                'transcr': transcr,
                'image_name': image_name}

    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16, 16], dtype=torch.float32)

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                               dtype=torch.float32)

        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])

            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']

                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]

            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref  # invert the image

        return {'img': imgs, 'style': style_ref, 'content': content_ref, 'wid': wid,
                'target': target, 'target_lengths': target_lengths, 'image_name': image_name}


"""random sampling of style images during inference"""


class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, ref_num) -> None:
        self.style_path = style_path

        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num

    def __len__(self):
        return self.ref_num

    def get_style_ref(self, wr_id):  # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)

            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image / 255.0

        return style_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            wid = idx
            batch.append({'style': style_ref, 'wid': wid})

        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                               dtype=torch.float32)

        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']

                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]

                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)

        return {'style': style_ref, 'wid': wid_list}


class Some_StyleIAMDataset:
    def __init__(self, style_path, ref_num, wid_num=None, fixed_height=64):
        self.style_path = style_path
        self.author_id = os.listdir(self.style_path)
        self.style_len = style_len  # 最大宽度
        self.fixed_height = fixed_height  # 固定高度
        self.ref_num = ref_num
        self.wid_num = wid_num if wid_num is not None else self.author_id

        # print(self.wid_num)
        # print(self.style_path)

    def __len__(self):
        return self.ref_num

    def get_style_ref(self, wr_id):
        # 随机选择满足宽度大于128的图像
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for style_ref in style_list:
            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)

            # 调整高度为固定值，宽度比例缩放
            style_image = cv2.resize(style_image, (style_image.shape[1] * self.fixed_height // style_image.shape[0],
                                                   self.fixed_height))

            if style_image.shape[1] > 128:
                break

        style_image = style_image / 255.0

        return style_image

    def __getitem__(self, _):
        batch = []
        for idx in self.wid_num:
            style_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0).to(torch.float32)

            batch.append({'style': style_ref, 'wid': idx})

        # 计算最大宽度，并填充样本
        max_width = min(self.style_len, max([item['style'].shape[2] for item in batch]))
        style_ref = torch.ones([len(batch), 1, self.fixed_height, max_width], dtype=torch.float32)

        wid_list = []
        for idx, item in enumerate(batch):
            current_width = min(item['style'].shape[2], max_width)
            style_ref[idx, :, :, :current_width] = item['style'][:, :, :current_width]

            wid_list.append(item['wid'])

        return {'style': style_ref, 'wid': wid_list}


class Random_WID_IAMDataset(IAMDataset):
    def __init__(self, style_path, style_len=128, ref_num=None) -> None:
        """
        :param style_path: 样式图像路径
        :param style_len: 样式图像最大宽度
        :param ref_num: 每次随机抽取的数量（默认使用 wid 目录下的所有图像数量）
        """
        self.style_path = style_path
        self.author_id = sorted(os.listdir(os.path.join(self.style_path)), key=lambda x: int(x))  # 获取所有 writer id (wid)
        self.style_len = style_len
        self.ref_num = ref_num  # 每次返回的样本数量（默认为 None）
        print(self.author_id)

    def __len__(self):
        return len(self.author_id)

    def get_style_ref(self, wr_id, num_samples):
        """
        获取指定  id (wid) 下的样式图像图像。
        随机抽取 num_samples 张图像，可重复抽取。
        """
        style_list = os.listdir(os.path.join(self.style_path, wr_id))  # 当前 wid 下的所有图像
        if self.ref_num is None:  # 如果用户没有输入 ref_num，默认使用当前 wid 的图像数量
            num_samples = len(style_list)
        else:
            num_samples = self.ref_num

        random.shuffle(style_list)  # 打乱图像列表
        selected_styles = random.choices(style_list, k=num_samples)  # 随机放回抽取

        style_images = []

        for style_ref in selected_styles:
            # 读取样式图像
            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)

            # 归一化处理
            style_image = style_image / 255.0

            style_images.append(style_image)

        return style_images

    def __getitem__(self, idx):
        """
        对每个 writer id 返回同一 wid 下的随机抽取样式图像图像。
        """
        wr_id = self.author_id[idx]  # 获取当前 writer id
        style_images = self.get_style_ref(wr_id, self.ref_num)

        widths = [img.shape[1] for img in style_images]
        max_width = max(widths)  # 重新动态计算所有图像的最大宽度
        # 确定样式图像的最大宽度

        # 初始化样式图像的 Tensor
        style_ref = torch.ones((len(style_images), 1, style_images[0].shape[0], max_width), dtype=torch.float32)

        # 填充 Tensor
        for i, style in enumerate(style_images):
            style_ref[i, :, :, :style.shape[1]] = torch.from_numpy(style).unsqueeze(0).to(torch.float32)

        # 返回样式图像和 writer id
        return {
            'style': style_ref,  # 样式图像
            'wid': wr_id  # writer id
        }


class ContentData(IAMDataset):
    def __init__(self, content_type='unifont') -> None:
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)

    def get_content(self, label):
        # 检查label类型，如果是字符串，直接处理；如果是列表，批量处理
        if isinstance(label, str):
            # 单个label的处理
            word_arch = [self.letter2index[i] for i in label]
            content_ref = self.con_symbols[word_arch]
            content_ref = 1.0 - content_ref
            return content_ref.unsqueeze(0)  # 返回的格式是单个样本的content_ref
        elif isinstance(label, list):
            # 对于列表中的每个label，批量处理
            content_refs = []
            max_length = 0
            # 先获取所有 content_ref 的最大长度
            for single_label in label:
                word_arch = [self.letter2index[i] for i in single_label]
                content_ref = self.con_symbols[word_arch]
                content_ref = 1.0 - content_ref
                content_refs.append(content_ref.unsqueeze(0))
                max_length = max(max_length, content_ref.shape[1])  # 假设在第1维是序列长度

            # 对所有 content_ref 进行填充，使其形状一致
            # print(max_length)
            padded_content_refs = []
            for content_ref in content_refs:
                # print(content_ref.shape)
                padding_size = max_length - content_ref.shape[1]
                padded_content_ref = F.pad(content_ref, (0, 0, 0, 0, 0, padding_size), value=1.0)  # 在最后一维填充
                # print(padded_content_ref.shape)
                padded_content_refs.append(padded_content_ref)

            # 拼接所有填充后的张量

            return torch.cat(padded_content_refs, dim=0)
        else:
            raise TypeError("Label must be a string or a list of strings.")
