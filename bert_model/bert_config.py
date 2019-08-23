# -*-coding:utf-8-*-
DEFAULT_CONFIGS = {
    "data_dir": "../data",
    "bert_model": "../bert_base_model",
    "task_name": "MyPro",
    "output_dir": "../bert_snapshot",
    "model_save_path": "../bert_snapshot/bert_model.pkl",
    "max_seq_length": 128,  # 字符串最大长度
    "do_train": True,  # 训练模式
    "do_eval": True,  # 验证模式
    "do_lower_case": False,  # 英文字符的大小写转换，对于中文来说没啥用
    "train_batch_size": 4,  # 训练时batch大小
    "eval_batch_size": 4,  # 验证时batch大小
    "learning_rate": 5e-5,  # Adam初始学习步长
    "num_train_epochs": 100.0,  # 训练的epochs次数
    "warmup_proportion": 0.1,  # Proportion of training to perform linear learning rate warmup for.
    "no_cuda": False,  # 用不用CUDA
    "local_rank": -1,  # local_rank for distributed training on gpus.
    "seed": 777,  # 初始化时的随机数种子
    "gradient_accumulation_steps": 1,  # Number of updates steps to accumulate before performing a backward/update pass.
    "optimize_on_cpu": False,  # Whether to perform optimization and keep the optimizer averages on CPU.
    "fp16": False,  # Whether to use 16-bit float precision instead of 32-bit.
    "loss_scale": 128,  # Loss scaling, positive power of 2 values can improve fp16 convergence.

}
