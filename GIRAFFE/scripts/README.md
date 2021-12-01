# Read Me about GIRAFFE论文阅读报告

## 前言

GIRAFFE论文荣获CVPR2021 Best Paper奖项，本次我的计算机视觉课程报告就是以阅读GIRAFFE论文为主，并尝试将其复现及应用。

托论文作者的开源福利，阅读论文的同时可以参考其[开源代码](https://github.com/autonomousvision/giraffe)，方便理解GIRAFFE的思想及其模型架构。复现代码时，我按照自己的理解写了较为详细的注释。

阅读报告本体使用CVPR2022官方LaTeX模板，在[Overleaf](https://www.overleaf.com)上加入UTF8宏包中文支持后使用`xelatex`作为编译引擎即可编译模板。

## 工程目录详解

- configs
  包含训练使用的配置文件和环境配置文件
  - `default.yaml`
    默认配置文件，是其他配置文件的父文件
  - `giraffe_on_\<datasetname\>.yaml`
    使用非默认数据集时的子配置文件，会覆盖默认配置的相关选项
  - `env.yml`
    构建环境使用的conda环境配置
- discriminator
  GAN的判别器网络部分，GIRAFFE使用的是Deep Convolutional Discriminator(from DCGAN)作为判别器网络
- Evalutation
  通用生成式模型评估部分。由于当下评估GAN的性能时通常采用FID分数(Frechet Inception Distance score[^1])计算两个多维变量分布之间的距离(即真实图像与生成图像在各自分布上的距离)，而FID分数由Inception Net-V3的最后一个池化层(**非输出层**)计算得到，故此文件夹包含:
  - `inception.py`
    Inception Net-V3的网络架构
  - `fid_score.py`
    计算FID分数的接口
  - `calc_gt_fid.py`
    训练GIRAFFE前用来计算原始数据集ground-truth FID分数

- GIRAFFE
  GIRAFFE模型(GAN的生成器部分)架构，由解码器`decoder.py`、2D神经渲染器`neural_renderer.py`、碰撞箱生成器`bounding_box_generator.py`组成，最后在`generator.py`中组装为整体生成器模型
  - scripts 进行训练和渲染的脚本所在的文件夹，包括`train.py`和`render.py`，见[训练部分](#训练)
- tools
  - `checkpoints.py`
    负责保存或加载模型检查点
  - `config.py`
    负责调用不同模型的`Config`类，处理包括读取配置文件、加载数据集、获得模型实例、模型训练器实例、模型渲染器实例等功能，一般由`train.py`调用

## 训练与渲染

运行代码时，需要将各文件夹所在的根目录作为工作目录，例如以`GIRAFFE`文件夹所在的目录作为当前工作目录，调用训练或渲染脚本时以模块的形式调用位于目录深层的相应脚本文件。

根据选择的数据集，首先要计算其FID(Frechet Inception Distance)分数作为Ground Truth，训练时需要gt-FID分数去评估生成器的阶段性效果

### 评估数据集的ground-truth FID-score

```shell
python -m Evalutation.calc_gt_fid <path/to/dataset> --img-size 64 --regex True --gpu 0 --out-file <path/to/out_file> 
```

### 训练

参考父配置文件`default.yaml`和子配置文件示例`giraffe_on_celebA.yaml`，注意要更改`[data][path]`为数据集所在路径和`[data][fid_file]`为上一步中得到的评估文件的路径

```shell
python -m GIRAFFE.scripts.train configs/giraffe_on_<dataset_name>.yaml
```

### 渲染

训练完毕后，模型的最佳检查点会被保存到`<outdir>/model_best.pt`，其中`out_dir`为yaml文件中的字段`giraffe_on_<dataset_name>.yaml[training][out_dir]`

```shell
python -m GIRAFFE.scripts.render configs/giraffe_on_<dataset_name>.yaml
```

运行完毕后，可以在`<out_dir>/rendering`目录下查看训练好的GIRAFFE模型的图像合成效果

### 渲染方式举例

在`GIRAFFE/renderer::render_full_visualization()`函数中，根据从配置文件读取的`render_program`字段中包含的渲染方式字符串，会生成其对应输出。

比如在CompCars数据集上(该数据集主要用于精准车辆识别)，每张图片只包含一辆汽车，使用GIRAFFE在其上进行训练后，可以合成包含多辆汽车对象的图片。

```yaml
# 在giraffe_on_compcars.yaml中编辑，追加“添加对象”渲染方式
rendering:
  render_program: [..., "render_add_cars"]
```

在渲染的输出文件夹中可以看到:

![渲染输出目录](https://i.loli.net/2021/12/01/PF9iAtSjOEua7fp.png)

为直观起见，取视频add_cars.mp4的一帧

![增加车辆数目](https://i.loli.net/2021/12/01/ZiqauJdoy4s1OtK.png)

## 参考

[^1]:Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017
