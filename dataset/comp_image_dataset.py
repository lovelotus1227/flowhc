from itertools import product
from random import choice
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop
import os
import json

# 保持原图像预处理配置，与论文ViT-B编码器兼容
BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    """图像预处理：兼容论文的ViT-B输入要求，保持原逻辑不变
    输入：任意尺寸、任意通道的 PIL 图像（维度 H×W×C）
    输出：(3, 224, 224) 的 FloatTensor，像素值分布在 [-2, 2]，可直接输入 ViT-B 模型"""
    if imagenet:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        # 将多个图像变换操作「串联」成一个有序的操作序列
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    return transform


class ImageLoader:
    """图像加载器：适配C-GQA的图像路径结构"""
    def __init__(self, root):
        self.img_dir = root  # 输入为图像根目录（如 /data/C-GQA/images）

    def __call__(self, img_name):
        """加载单张图像：兼容C-GQA的图像命名格式"""
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            # 兼容不同后缀（如jpg/png）
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        return img


class CompositionImageDataset(Dataset):
    """适配C-GQA的图像数据集类：兼容原模型的输入格式和论文复现需求"""
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            same_prim_sample=False,
    ):
        self.root = root
        self.phase = phase  # train/val/test
        self.split = split
        self.open_world = open_world
        self.same_prim_sample = same_prim_sample

        # 路径配置：严格适配C-GQA的目录结构
        self.img_root = os.path.join(root, 'images')  # 图像存储目录
        self.split_root = os.path.join(root, 'compositional-split-natural')  # 标签文件目录
        self.metadata_path = os.path.join(root, f'metadata_{split}.t7')  # 元数据路径

        # 图像预处理和加载器
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.img_root)

        # 解析标签对（属性-物体）：兼容论文的组合式标签格式
        self.attrs, self.objs, self.pairs, \
        self.train_pairs, self.val_pairs, \
        self.test_pairs = self.parse_split()

        # 开放世界模式：生成所有可能的属性-物体组合
        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        # 加载训练/验证/测试数据
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.data = self.train_data if phase == 'train' else (self.val_data if phase == 'val' else self.test_data)

        # 标签映射：与原模型的索引体系保持一致
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        # 打印数据集信息，方便调试
        print(f'# Dataset: C-GQA | Split: {split} | Phase: {phase}')
        print(f'# train pairs: {len(self.train_pairs)} | # val pairs: {len(self.val_pairs)} | # test pairs: {len(self.test_pairs)}')
        print(f'# train images: {len(self.train_data)} | # val images: {len(self.val_data)} | # test images: {len(self.test_data)}')
        print(f"开放世界:{self.open_world}")
        # 训练集组合对映射：用于损失计算
        self.train_pair_to_idx = {(pair): idx for idx, pair in enumerate(self.train_pairs)}

        # 开放世界相关配置（与原逻辑一致）
        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        # 同原语采样配置（用于辅助损失）
        if self.phase == 'train' and self.same_prim_sample:
            self.same_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            self.same_obj_diff_attr_dict = {pair: list() for pair in self.train_pairs}
            for i_sample, sample in enumerate(self.train_data):
                sample_attr, sample_obj = sample[1], sample[2]
                for pair_key in self.same_attr_diff_obj_dict.keys():
                    if (pair_key[1] == sample_obj) and (pair_key[0] != sample_attr):
                        self.same_obj_diff_attr_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] == sample_attr):
                        self.same_attr_diff_obj_dict[pair_key].append(i_sample)

    def parse_split(self):
        """解析C-GQA的标签对文件：支持txt/json两种格式，兼容论文数据集"""
        def parse_pairs(file_path):
            # 兼容txt（每行"attr obj"）和json（列表格式）
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    pairs = f.read().strip().split('\n')
                    pairs = [t.split() for t in pairs if t.strip()]  # 按空格分割
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    items = json.load(f)
                    pairs = [[item['attribute'], item['object']] for item in items]  # 适配C-GQA的json字段
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs) if pairs else ([], [])
            return attrs, objs, pairs

        # 加载训练/验证/测试的标签对文件
        tr_pairs_path = os.path.join(self.split_root, 'train_pairs.txt')
        vl_pairs_path = os.path.join(self.split_root, 'val_pairs.txt')
        ts_pairs_path = os.path.join(self.split_root, 'test_pairs.txt')

        # 兼容json格式（若txt不存在则尝试json）
        for path in [tr_pairs_path, vl_pairs_path, ts_pairs_path]:
            if not os.path.exists(path):
                json_path = path.replace('.txt', '.json')
                if os.path.exists(json_path):
                    path = json_path

        tr_attrs, tr_objs, tr_pairs = parse_pairs(tr_pairs_path)
        vl_attrs, vl_objs, vl_pairs = parse_pairs(vl_pairs_path)
        ts_attrs, ts_objs, ts_pairs = parse_pairs(ts_pairs_path)

        # 合并所有属性/物体/组合对并排序
        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        """加载训练/验证/测试数据：兼容t7和json格式，适配C-GQA的标注"""
        train_data, val_data, test_data = [], [], []

        # 优先加载t7格式（原逻辑），若不存在则加载json格式
        if os.path.exists(self.metadata_path):
            data = torch.load(self.metadata_path)
            for instance in data:
                image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']
                # 过滤无效标注
                if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                    continue
                data_i = [image, attr, obj]
                if settype == 'train':
                    train_data.append(data_i)
                elif settype == 'val':
                    val_data.append(data_i)
                else:
                    test_data.append(data_i)
        else:
            # 加载json格式标注（C-GQA常用）
            json_paths = {
                'train': os.path.join(self.split_root, 'train_pairs.json'),
                'val': os.path.join(self.split_root, 'val_pairs.json'),
                'test': os.path.join(self.split_root, 'test_pairs.json')
            }
            for settype, json_path in json_paths.items():
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r') as f:
                    items = json.load(f)
                    for item in items:
                        image = item['img_name']
                        attr = item['attribute']
                        obj = item['object']
                        if attr == 'NA' or (attr, obj) not in self.pairs:
                            continue
                        data_i = [image, attr, obj]
                        if settype == 'train':
                            train_data.append(data_i)
                        elif settype == 'val':
                            val_data.append(data_i)
                        else:
                            test_data.append(data_i)

        return train_data, val_data, test_data

    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        """同原语不同组合采样：用于辅助损失计算，保持原逻辑不变"""
        if phase == 'attr':
            candidate_list = self.same_attr_diff_obj_dict.get((label_A, label_B), [])
        else:
            candidate_list = self.same_obj_diff_attr_dict.get((label_B, label_A), [])
        if len(candidate_list) != 0:
            idx = choice(candidate_list)
            mask = 1
        else:
            idx = choice(list(range(len(self.data))))
            mask = 0
        return self.data[idx], mask

    def __getitem__(self, index):
        """
        核心：返回格式与原视频数据集完全一致，保证模型兼容
        返回：[img, attr_idx, obj_idx, pair_idx, (可选：辅助采样数据)]
        """
        image_name, attr, obj = self.data[index]
        # 加载并预处理图像
        img = self.loader(image_name)
        img = self.transform(img)  # 输出形状：[3, 224, 224]（ViT-B标准输入）

        if self.phase == 'train':
            # 训练阶段：返回格式与原视频数据集一致
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
            # 同原语采样：添加辅助训练数据
            if self.same_prim_sample:
                [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
                [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
                # 加载并预处理辅助图像
                same_attr_img = self.transform(self.loader(same_attr_image))
                same_obj_img = self.transform(self.loader(same_obj_image))
                # 补充辅助数据（与原逻辑一致）
                data += [
                    same_attr_img, self.attr2idx[same_attr], self.obj2idx[diff_obj],
                    self.train_pair_to_idx.get((same_attr, diff_obj), -1), same_attr_mask,
                    same_obj_img, self.attr2idx[diff_attr], self.obj2idx[same_obj],
                    self.train_pair_to_idx.get((diff_attr, same_obj), -1), same_obj_mask
                ]
        else:
            # 验证/测试阶段：返回格式与原视频数据集一致
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        """数据集长度：图像样本数量"""
        return len(self.data)