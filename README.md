# AngryScene：Jittor风景生成赛（Pix2PixHD）

本仓库代码应用于[第二届计图人工智能挑战赛赛题一：风景图片生成赛题](https://www.educoder.net/competitions/index/Jittor-3)

风景图片生成赛道的任务是基于语义分割图生成有意义、高质量的风景图片，本仓库基于Pix2PixHD实现。

总分：0.4569

排名：A榜19名

#### Results
　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p4g5qu3wj20e80ao3z6.jpg" alt="2095304_642b186684_b" style="zoom:50%;" /></td><td><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p4glaetyj20e80aojrn.jpg" alt="2095304_642b186684_b" style="zoom:50%;"></td>
     <td style="padding: 6px"><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p4h3vzdcj20e80ao3yt.jpg" alt="2095304_642b186684_b" style="zoom:50%;" /></td>
		</tr>
    <tr><td></td><td style="text-align:center"><strong>图 3  Pix2Pix HD-测试集生成图片</strong></td><td></td></tr>
	</tbody>
</table>

　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p4hh1dpuj20e80aoaad.jpg" alt="2095304_642b186684_b" style="zoom:50%;" /></td><td><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p32mebccj20e80aodg2.jpg" alt="2095304_642b186684_b" style="zoom:50%;"></td>
     <td style="padding: 6px"><img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h3p4i076dtj20e80aoq38.jpg" alt="2095304_642b186684_b" style="zoom:50%;" /></td>
		</tr>
    <tr><td></td><td style="text-align:center"><strong>图 3  Pix2Pix HD-测试集生成图片</strong></td><td></td></tr>
	</tbody>
</table>

#### Installation

安装依赖环境

```shell
pip install -r requirements.txt
```

#### Dependencies

- Linux
- Python=3.7
- Jittor >= 1.3.0
- NVIDIA GPU + CUDA cuDNN

#### File Tree

```
.
├── README.md                   
├── Pix2PixHD
│   ├── data										// 处理数据
│   ├── models                	// 网络结构
│   ├── options        					// 选项
│   ├── util                		// 常用工具
│   ├── encode_features.py      
│   ├── procompute_feature_maps.py
│   ├── resize_result.py
│   ├── resize.py      
│   ├── run_engine.py
│   ├── test.py
│   └── train.py
├── Pix2Pix
├── StyleGAN
├── results
└── datasets
```

#### Usage

- Preprocess

  ```shell
  cd Pix2PixHD
  python resize.py
  ```

- Train

  ```shell
  cd Pix2PixHD
  python train.py --no_instance
  ```

  可选参数：

  ```shell
  --dataroot //更改数据集根目录
  --no_flip //不进行以数据增强为目的的图片的水平翻转
  ```

- Test

  ```shell
  cd Pix2PixHD
  python test.py --no_instance
  python result_resize.py
  ```

#### Dataset

本仓库使用的训练数据可在[此处](https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1)下载，测试数据在[此处](https://cloud.tsinghua.edu.cn/f/70195945f21d4d6ebd94/?dl=1)下载。

#### Acknowledgements

本仓库的实现参考了以下仓库：[pix2pix-baseline](https://github.com/Jittor/JGAN/tree/master/competition#%E8%B5%9B%E9%A2%98%E4%B8%80%E9%A3%8E%E6%99%AF%E5%9B%BE%E7%89%87%E7%94%9F%E6%88%90%E8%B5%9B%E9%A2%98)、[pix2pixHD](https://github.com/NVIDIA/pix2pixHD)、[styleGAN-jittor](https://github.com/xUhEngwAng/StyleGAN-jittor)、[styleMapGAN](https://github.com/naver-ai/StyleMapGAN)、[DiffAug](https://github.com/mit-han-lab/data-efficient-gans)。

其中比赛实际使用的pix2pixHD模型经过了pytorch-jittor转换，整体思路保持一致，其余模型大多起到参考作用。