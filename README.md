<img src="./rt1.png" width="450px"></img>

## Robotic Transformer - Pytorch

Implementation of <a href="https://ai.googleblog.com/2022/12/rt-1-robotics-transformer-for-real.html">RT1 (Robotic Transformer)</a>, from the Robotics at Google team, in Pytorch. As well as the training script for the model.

## Install

```bash
$ pip install robotic-transformer-pytorch
```

## Usage

```python
import torch
from robotic_transformer_pytorch import MaxViT, RT1

vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 2, 5, 2),
    window_size = 7,
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

train_logits = model(video, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)

# after much training

model.eval()
eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3

```

### Training

To train the model, you can use the `train.py` script. Here is an example of how to run the training:

```bash
python train.py
```

You can also customize the training configuration by modifying the `config` dictionary in `train.py`.

```python
if __name__ == "__main__":
    set_log_level("INFO")
    torch.set_float32_matmul_precision("medium")
    # 训练配置
    config = {
        "data_dir": "/data/shared_folder/h5_ur_1rgb/raw_dataset",
        "output_dir": "./outputs",
        "batch_size": 8,
        "num_workers": 4,
        "max_epochs": 100,
        "gpus": 1,
        "resume_from_checkpoint": None,  # 或者指定检查点路径
    }

    train(**config)
```

### Dataset

The `dataset.py` script provides a `RobotMindDataset` class for loading and processing the `RoboMIND` dataset. Here is an example of how to use it:

```python
from dataset import create_dataloader

data_dir = "/data/shared_folder/h5_ur_1rgb/raw_dataset"
train_loader, val_loader = create_dataloader(
    data_dir=data_dir,
    split="merge",
    window_size=6,
    val_ratio=0.2,
    batch_size=8,
    num_workers=4,
)

for batch in train_loader:
    print(batch)
```


## Todo

- [ ] valuating the model trained by the `train.py` script



## Appreciation

- [lucidrains](https://github.com/lucidrains): for the implementation of the `RT1` model in Pytorch


## Citations

```bibtex
@inproceedings{rt12022arxiv,
    title    = {RT-1: Robotics Transformer for Real-World Control at Scale},
    author   = {Anthony Brohan and Noah Brown and Justice Carbajal and  Yevgen Chebotar and Joseph Dabis and Chelsea Finn and Keerthana Gopalakrishnan and Karol Hausman and Alex Herzog and Jasmine Hsu and Julian Ibarz and Brian Ichter and Alex Irpan and Tomas Jackson and  Sally Jesmonth and Nikhil Joshi and Ryan Julian and Dmitry Kalashnikov and Yuheng Kuang and Isabel Leal and Kuang-Huei Lee and  Sergey Levine and Yao Lu and Utsav Malla and Deeksha Manjunath and  Igor Mordatch and Ofir Nachum and Carolina Parada and Jodilyn Peralta and Emily Perez and Karl Pertsch and Jornell Quiambao and  Kanishka Rao and Michael Ryoo and Grecia Salazar and Pannag Sanketi and Kevin Sayed and Jaspiar Singh and Sumedh Sontakke and Austin Stone and Clayton Tan and Huong Tran and Vincent Vanhoucke and Steve Vega and Quan Vuong and Fei Xia and Ted Xiao and Peng Xu and Sichun Xu and Tianhe Yu and Brianna Zitkovich},
    booktitle = {arXiv preprint arXiv:2204.01691},
    year      = {2022}
}
```

```bibtex
@inproceedings{Tu2022MaxViTMV,
    title   = {MaxViT: Multi-Axis Vision Transformer},
    author  = {Zhengzhong Tu and Hossein Talebi and Han Zhang and Feng Yang and Peyman Milanfar and Alan Conrad Bovik and Yinxiao Li},
    year    = {2022}
}
```

```bibtex
@misc{peebles2022scalable,
    title   = {Scalable Diffusion Models with Transformers},
    author  = {William Peebles and Saining Xie},
    year    = {2022},
    eprint  = {2212.09748},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
