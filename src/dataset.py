import ast
import torch
import time
import json
import numpy as np

from pathlib import Path
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.models.clip import tokenize
from src.prompt_matcher import match_prompt
from src.models.vggish import wavfile_to_examples

# 问题类型到索引的映射字典
# 用于将问题类型（如音频计数、视觉定位等）转换为数字标签
qtype2idx = {
    'Audio': {'Counting': 0, 'Comparative': 1},
    'Visual': {'Counting': 2, 'Location': 3},
    'Audio-Visual': {'Existential': 4, 'Counting': 5, 'Location': 6,
                     'Comparative': 7, 'Temporal': 8}
}


FILE = Path(__file__).resolve() # 获取当前文件的绝对路径
ROOT = FILE.parents[1] # 获取项目的根目录 (假设dataset.py在src目录下，则根目录是src的上级目录)

# 定义AVQA数据集类，继承自torch.utils.data.Dataset
class AVQA_dataset(Dataset):

    def __init__(self, 
                 config: dict, # 传入的配置字典，包含了数据路径、模型参数等信息
                 mode: str, # 数据集模式，例如 'train', 'val', 'test'
                 transform: transforms.Compose = None, # 图像预处理的转换操作
    ):
        self.mode = mode  # 存储数据集模式
        self.config = config # 存储配置字典
        self.type = config.type # 存储配置中的类型信息 (例如 'qa-tiger')
        self.root = config.data.root # 数据集的根目录路径
        
        # 根据配置，构建音频、视频、patch、问题和提示特征的完整路径
        # 如果配置中对应的特征路径为None，则相应的特征路径也为None
        self.audio_feat = (ROOT / self.root / config.data.audio_feat).as_posix() \
            if config.data.audio_feat is not None else None
        self.video_feat = (ROOT / self.root / config.data.video_feat).as_posix() \
            if config.data.video_feat is not None else None
        self.patch_feat = (ROOT / self.root / config.data.patch_feat).as_posix() \
            if config.data.patch_feat is not None else None
        self.quest_feat = (ROOT / self.root / config.data.quest_feat).as_posix() \
            if config.data.quest_feat is not None else None
        self.prompt_feat = (ROOT / self.root / config.data.prompt_feat).as_posix() \
            if config.data.prompt_feat is not None else None
        
        self.tokenizer = tokenize # 使用src.models.clip中的tokenize函数作为文本tokenizer
        self.bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased') # 加载BERT的tokenizer，可能用于其他文本处理或对比实验
        self.size = config.data.img_size # 图像的目标尺寸
        self.sample_rate = config.data.frame_sample_rate # 视频帧的采样率
        
        # 根据模式（train/val/test）获取对应的标注文件名
        annot_key = f'{self.mode}_annot'
        # 从配置中动态获取标注文件名 (例如 config.data.train_annot)
        annot_filename = eval(f"self.config.data.{annot_key}")
        annot_path = ROOT / self.root / annot_filename # 构建标注文件的完整路径
        
        # 读取并处理标注文件
        with open(file=annot_path.as_posix(), mode='r') as f:
            # 调用question_process方法对从json加载的样本进行预处理
            self.samples = self.question_process(json.load(f))
        
        # 获取答案词汇表和最大问题长度等信息
        ans_quelen_info = self.get_max_question_length()
        self.answer_to_ix = ans_quelen_info['ans2ix'] # 答案到索引的映射
        self.max_que_len = ans_quelen_info['max_que_len'] # 数据集中问题的最大长度
        self.config.num_labels = len(self.answer_to_ix) # 更新配置中的标签数量（答案类别数）
        
        # 提取所有唯一的视频ID列表 (这部分代码似乎未完成，video_name未被使用)
        video_list = []
        for sample in self.samples:
            video_name = sample['video_id'] # 获取样本中的视频ID
            if video_name not in video_list:
                video_list.append(video_name) # 如果视频ID不在列表中，则添加
                
        self.video_list = video_list # 存储唯一的视频ID列表
        self.video_len = 60 * len(video_list) # 视频总长度的估计值 (假设每个视频60帧，可能不准确)
        self.cache = {} # 初始化缓存字典，可能用于存储已加载的特征以加速访问
        
        # 设置图像转换操作
        # 如果外部传入了transform，则使用外部的；否则，创建默认的transform
        self.transform = transform if transform is not None \
            else transforms.Compose([
                    transforms.Resize((self.size, self.size)), # 调整图像大小
                    transforms.ToTensor(), # 将PIL Image或numpy.ndarray转换为torch.Tensor
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, # 标准化图像
                                         std=IMAGENET_DEFAULT_STD),
                ])

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.samples)
    
    # 加载单个样本的数据
    def load_samples(self, sample):
        # 问题和答案的预处理
        # 将答案文本转换为对应的索引标签
        labels = torch.tensor(data=[self.answer_to_ix[sample['anser']]], dtype=torch.long)
        # 将问题类型字符串（例如 "['Audio', 'Counting']"）转换为实际的Python列表
        ques_type_str = sample['type']
        # 使用ast.literal_eval安全地评估字符串，避免潜在的安全风险
        ques_type_list = ast.literal_eval(ques_type_str) 
        # 将问题类型转换为数字标签
        qtype_label = torch.tensor([qtype2idx[ques_type_list[0]][ques_type_list[1]]], dtype=torch.long)
        
        # 加载问题特征
        if self.quest_feat is not None:
            # 如果配置了预计算的问题特征路径，则加载.npy文件
            quest_id = sample['question_id']
            quest = np.load(Path(self.quest_feat) / f'{int(quest_id)}.npy')
            # 同样加载预计算的提示特征 (如果存在)
            prompt = np.load(Path(self.prompt_feat) / f'{int(quest_id)}.npy')
        else:
            # 如果没有预计算的问题特征，则在线进行tokenize
            question = sample['question_content']
            quest = self.tokenizer(question, truncate=True).squeeze() # 使用CLIP tokenizer
            # prompt = self.tokenizer(sample['qprompt'], truncate=True).squeeze() # 对问题提示进行tokenize
        
        # 加载视频帧/特征
        name = sample['video_id'] # 视频文件名或ID
        patch = None # 初始化patch特征为None
        if self.video_feat is not None:
            # 如果配置了预计算的视频特征，则加载.npy文件
            video = np.load(Path(self.video_feat) / f'{name}.npy')
            video = torch.from_numpy(video)[::self.sample_rate] # 转换为Tensor并按采样率采样
            if self.patch_feat is not None:
                # 如果配置了预计算的patch特征路径，则加载
                patch = np.load(Path(self.patch_feat) / f'{name}.npy')
                patch = torch.from_numpy(patch)[::self.sample_rate]
            # else: # 这部分逻辑看起来不完整
                # patch = None 
        else:
            # 如果没有预计算的视频特征，则从原始帧文件加载
            frame_dir = ROOT / self.root / self.config.data.frames_dir / name
            # 获取帧文件路径列表，排序并取前60帧 (可能为了处理某些视频帧数超标的情况)
            frame_path = sorted(list(frame_dir.glob('*.jpg')))[:60] 
            frame_path = frame_path[::self.sample_rate] # 按采样率采样
            # 此处缺少将frame_path中的图像加载并转换为Tensor的逻辑，例如使用self.transform
            # video = torch.stack([self.transform(Image.open(p)) for p in frame_path]) # 示例逻辑
            video = [] # 应该填充实际的视频数据

        
        # 加载音频特征/数据
        if self.audio_feat is not None:
            # 如果配置了预计算的音频特征路径，则加载.npy文件
            audio = np.load(Path(self.audio_feat) / f'{name}.npy')
            audio = torch.from_numpy(audio) # 转换为Tensor
        else:
            # 如果没有预计算的音频特征，则从原始音频文件加载并提取特征
            # audio_path = ROOT / self.root / self.config.data.audios_dir / f'{name}.wav' # 假设是.wav格式
            # audio = wavfile_to_examples(audio_path.as_posix()) # 使用vggish提取特征的示例
            audio = [] # 应该填充实际的音频数据
        
        # 组织返回的数据字典
        data = {
            'quest': quest, # 问题特征或tokenized问题
            # 'prompt': prompt, # 提示特征或tokenized提示
            'type': ques_type_list, # 问题类型列表 (例如 ['Audio', 'Counting'])
            'label': labels, # 答案的数字标签
            'qtype_label': qtype_label, # 问题类型的数字标签
            'video': video, # 视频帧特征或Tensor
            'audio': audio, # 音频特征或Tensor
            'name': name, # 视频/样本名称
        }
        if patch is not None:
            data['patch'] = patch # 如果存在patch特征，则添加到字典中
            
        return data
    
    # PyTorch Dataset类的核心方法，根据索引获取一个样本
    def __getitem__(self, index):
        sample = self.samples[index] # 根据索引获取原始样本信息
        batch = self.load_samples(sample) # 调用load_samples方法加载和预处理数据
        print("Batch keys:", batch.keys())
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"{k} shape:", v.shape)
            else:
                print(f"{k}:", v)

        return batch

    # 对从JSON加载的原始样本列表进行预处理
    def question_process(self, samples):
        for index, sample in enumerate(samples):
            # 可以在这里添加对每个sample的特定处理逻辑
            # 例如，清洗文本、统一格式等
            # 目前这个函数没有做实际的修改，直接返回原始samples
            pass # 示例：sample['question_content'] = sample['question_content'].lower()
        return samples
    
    # 获取答案词汇表、问题最大长度等信息
    def get_max_question_length(self):
        ans_quelen_path = ROOT / self.root / self.config.data.ans_quelen
        if ans_quelen_path.exists():
            # 如果存在预计算的ans_quelen文件 (通常是answer2idx.json)
            with open(ans_quelen_path.as_posix(), 'r') as f:
                ans_quelen_info = json.load(f)
        else:
            # 如果不存在，则需要遍历数据集动态生成这些信息 (这部分逻辑未实现)
            # 这通常在第一次运行时执行，然后保存结果以备后续使用
            # 需要遍历self.samples来构建答案词汇表和计算最大问题长度
            # ans_to_ix = {}
            # max_len = 0
            # for sample in self.samples: # 假设self.samples此时已加载
            #     ans = sample['anser']
            #     if ans not in ans_to_ix:
            #         ans_to_ix[ans] = len(ans_to_ix)
            #     q_len = len(self.tokenizer(sample['question_content'])) # 示例：计算tokenized长度
            #     if q_len > max_len:
            #         max_len = q_len
            # ans_quelen_info = {'ans2ix': ans_to_ix, 'max_que_len': max_len}
            # with open(ans_quelen_path.as_posix(), 'w') as f:
            #     json.dump(ans_quelen_info, f)
            raise FileNotFoundError(f"Annotation file {ans_quelen_path} not found and dynamic creation is not fully implemented.")
        return ans_quelen_info
