# config 字典定义了整个实验的所有配置参数
config = dict(
	type='qa-tiger',  # 实验类型或模型名称标识
	seed=713,  # 随机种子，用于保证实验可复现性
	epochs=15,  # 训练的总轮数
	num_labels=42,  # 分类任务的类别数量 (答案的数量)
	log_interval=100,  # 每隔多少个 batch 打印一次日志
	output_dir= '/mnt/sda/shenhao/code/QA-TIGER/qa-tiger_clip_vitl14@336px',  # 训练输出（日志、模型权重等）的保存目录
	pretrained_weight="base",  # 可能用于加载预训练权重的标识
	weight='/mnt/sda/shenhao/code/QA-TIGER/qa-tiger_clip_vitl14@336px/2025-05-18-10-53-23_seed713/best.pt',
	data=dict(  # 数据相关的配置
		root='./data',  # 数据集的根目录
		img_size=336,  # 输入图像的尺寸 (ViT-L/14@336px 模型需要336x336的图像)
		batch_size=32,  # 训练时的批量大小
		eval_batch_size=32,  # 验证/测试时的批量大小
		num_workers=8,  # 数据加载时使用的工作进程数
		frame_sample_rate=1,  # 视频帧的采样率 (例如，每隔1帧取1帧)

		audios_dir='./raw_audios',  # 原始音频文件的存放目录 (如果直接从原始文件加载)
		frames_dir='./raw_frames',  # 原始视频帧的存放目录 (如果直接从原始帧加载)
		train_annot='/mnt/sda/shenhao/code/QA-TIGER/data/annots/music_avqa/music_avqa_train.json',  # 训练集标注文件的路径
		valid_annot='/mnt/sda/shenhao/code/QA-TIGER/data/annots/music_avqa/music_avqa_val.json',  # 验证集标注文件的路径
		test_annot='/mnt/sda/shenhao/code/QA-TIGER/data/annots/music_avqa/music_avqa_test.json',  # 测试集标注文件的路径
		test_annots=None,  # 可能用于多个测试集的标注文件列表
		ans_quelen='/mnt/sda/shenhao/code/QA-TIGER/data/annots/music_avqa/answer2idx.json',  # 答案到索引的映射文件，以及可能包含问题最大长度信息
		
		# precomputed features - 预计算特征的路径配置
		# 如果这些路径被设置，则直接加载预计算好的特征，否则会从原始数据进行在线提取或处理
		quest_feat=None,  # 预计算的问题特征路径 (例如，使用BERT或CLIP文本编码器提取的特征)
		audio_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/vggish',  # 预计算的音频特征路径 (例如，VGGish 特征)
		video_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/clip_feats/1fps',  # 预计算的视频帧级别特征路径 (例如，使用CLIP ViT-L/14@336px 提取的特征)
		patch_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/visual_tome14_60',  # 预计算的视频patch级别特征路径 (例如，使用ToMe处理后的视觉特征)
		prompt_feat=None,  # 预计算的提示特征路径 (如果使用了特定的prompt)
	),

	hyper_params=dict(  # 超参数配置
		gpus='0',  # 使用的GPU ID (例如 '0', '0,1')
		model_type="QA-TIGER_ViTL14@336px",  # 模型的具体类型或版本名
		model=dict(  # 模型架构相关的参数
			d_model=512,  # 模型内部的主要隐藏层维度
			video_dim=768,  # 输入视频特征的原始维度 (例如 CLIP ViT-L/14 输出维度为768)
			patch_dim=1024, # 输入patch特征的原始维度 (例如 ToMe ViT-L/14 输出维度为1024)
			quest_dim=512,  # 输入问题特征的原始维度 (如果使用预计算的问题特征)
			audio_dim=128,  # 输入音频特征的原始维度 (例如 VGGish 输出维度为128)
			topK=7,  # MoE (Mixture of Experts) 中的 topK 参数
			num_experts=7,  # MoE 中的专家数量
			encoder_type='ViT-L/14@336px',  # 使用的CLIP文本编码器类型
		),
		optim=dict(  # 优化器相关的参数
			lr=1e-4,  # 学习率
			encoder_lr=None,  # 特定编码器（如文本编码器）的学习率，如果为None则使用全局lr
			min_lr=1e-7,  # 学习率的最小值 (例如，用于学习率衰减)
			weight_decay=0,  # 权重衰减系数
			betas=(0.95, 0.999)  # AdamW等优化器的beta参数
		),
		sched=dict(  # 学习率调度器相关的参数
			name='StepLR',  # 调度器的名称 (例如 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau')
			mode='min',  # ReduceLROnPlateau模式 ('min' 或 'max')，监控指标是最小化还是最大化
			gamma=0.1,  # StepLR的衰减因子
			step_size=8,  # StepLR的衰减步长 (每隔多少个epoch衰减一次)
			factor=0.5,  # ReduceLROnPlateau的衰减因子
			patience=5,  # ReduceLROnPlateau的耐心值 (多少个epoch内指标没有改善则衰减学习率)
			verbose=True,  # 是否打印学习率调度信息
			warmup_epochs=2,  # 学习率预热的轮数
		),
	)
)