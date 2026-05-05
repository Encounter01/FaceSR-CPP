# FaceSR_CPP 数据处理模块结构与架构

本文档依据当前项目源码整理，重点对应：

- `include/utils/dataset.h`
- `src/utils/dataset.cpp`
- `include/utils/image_utils.h`
- `src/utils/image_utils.cpp`
- `src/trainer.cpp`
- `src/inference.cpp`

如果要插入 Word/论文正文，优先使用 `docs/DATA_PROCESSING_FIGURES_WORD.md` 中的“论文插图版”。该版本字号更大、比例更接近正方形，导出为 SVG 后插入 Word 不易模糊。

## 1. 数据处理模块总体架构

### 1.1 论文紧凑版

该版本压缩了节点文字，并采用多列分层布局，导出图片后更接近正方形，适合放入论文正文。

```mermaid
flowchart LR
    subgraph A["配置与输入"]
        direction TB
        Cfg["TrainConfig<br/>路径 / 尺寸 / batch / workers"]
        Const["constants<br/>格式 / 归一化 / 增强概率"]
        HR["HR 目录<br/>train / val"]
        LR["可选 LR 目录<br/>为空则在线生成"]
    end

    subgraph B["文件发现"]
        direction TB
        Scan["getImageFiles()<br/>扫描"]
        Filter["格式过滤<br/>jpg / png / bmp / tiff"]
        Sort["排序<br/>保证顺序稳定"]
    end

    subgraph C["样本构造"]
        direction TB
        LoadHR["读取 HR<br/>BGR -> RGB"]
        ResizeHR["HR resize<br/>256 x 256"]
        MakeLR{"LR 来源"}
        LoadLR["读取 LR<br/>64 x 64"]
        Down["HR 下采样<br/>Bicubic 64 x 64"]
    end

    subgraph D["增强与张量化"]
        direction TB
        Aug["同步增强<br/>翻转 / 旋转"]
        Norm["归一化<br/>uint8 -> float32<br/>[0,1]"]
        CHW["维度转换<br/>HWC -> CHW"]
        Ex["Example<br/>data=LR<br/>target=HR"]
    end

    subgraph E["批处理与训练"]
        direction TB
        Stack["Stack<br/>样本 -> batch"]
        Loader["DataLoader<br/>多线程 / 随机采样"]
        Batch["Batch<br/>LR [B,3,64,64]<br/>HR [B,3,256,256]"]
        Train["Generator / Loss<br/>验证 PSNR / SSIM"]
    end

    Cfg --> HR
    Cfg --> LR
    Const --> Filter
    HR --> Scan --> Filter --> Sort --> LoadHR --> ResizeHR --> MakeLR
    LR --> MakeLR
    MakeLR -- 存在 LR --> LoadLR
    MakeLR -- 在线生成 --> Down
    ResizeHR --> Down
    LoadLR --> Aug
    Down --> Aug
    ResizeHR --> Aug
    Aug --> Norm --> CHW --> Ex --> Stack --> Loader --> Batch --> Train
```

### 1.2 完整说明版

```mermaid
flowchart TB
    subgraph Config["配置层"]
        TrainConfig["TrainConfig<br/>train_hr_dir / train_lr_dir<br/>val_hr_dir / val_lr_dir<br/>hr_size=256 / lr_size=64<br/>batch_size / num_workers"]
        Constants["constants<br/>SUPPORTED_EXTENSIONS<br/>NORMALIZE_SCALE<br/>AUGMENT_*_PROB"]
    end

    subgraph IO["图像文件输入层"]
        HRDir["HR 图像目录<br/>data/train/hr 或 data/val/hr"]
        LRDir["可选 LR 图像目录<br/>为空则在线生成 LR"]
        ImageFiles["getImageFiles()<br/>扫描 / 过滤 / 排序"]
    end

    subgraph Dataset["样本构造层：FaceSRDataset"]
        LoadHR["loadImage(HR)<br/>cv::imread<br/>BGR -> RGB"]
        ResizeHR["resize HR<br/>256 x 256<br/>INTER_CUBIC"]
        LRMode{"是否配置 LR 目录<br/>且同名文件存在？"}
        LoadLR["loadImage(LR)<br/>resize 64 x 64"]
        GenLR["从 HR 在线下采样<br/>resize 64 x 64<br/>INTER_CUBIC"]
        Aug["applyAugmentation()<br/>同步水平翻转<br/>同步 90 度旋转"]
        ToTensor["matToTensor()<br/>uint8 -> float32<br/>[0,255] -> [0,1]<br/>HWC -> CHW"]
        Example["torch::data::Example<br/>data = LR Tensor<br/>target = HR Tensor"]
    end

    subgraph Loader["批处理层：LibTorch DataLoader"]
        Stack["Stack&lt;Example&gt;<br/>单样本堆叠为 batch"]
        DataLoader["make_data_loader()<br/>batch_size<br/>num_workers<br/>RandomSampler"]
        Batch["Batch<br/>LR: [B,3,64,64]<br/>HR: [B,3,256,256]"]
    end

    subgraph Train["训练/验证消费层"]
        Device["batch.to(device)<br/>CPU / CUDA"]
        Generator["Generator<br/>LR -> SR"]
        LossMetric["训练：Loss<br/>验证：PSNR / SSIM<br/>结果图保存"]
    end

    TrainConfig --> HRDir
    TrainConfig --> LRDir
    Constants --> ImageFiles
    HRDir --> ImageFiles --> LoadHR --> ResizeHR --> LRMode
    LRDir --> LRMode
    LRMode -- 是 --> LoadLR
    LRMode -- 否 --> GenLR
    ResizeHR --> GenLR
    LoadLR --> Aug
    GenLR --> Aug
    ResizeHR --> Aug
    Aug --> ToTensor --> Example
    Example --> Stack --> DataLoader --> Batch --> Device --> Generator --> LossMetric
```

## 2. FaceSRDataset 内部结构

### 2.1 论文紧凑版

```mermaid
flowchart LR
    subgraph S["内部状态"]
        direction TB
        Files["hr_images_<br/>HR 文件列表"]
        LRDir["lr_dir_<br/>可选 LR 路径"]
        Size["hr_size_ / lr_size_<br/>256 / 64"]
        AugCfg["aug_config_<br/>flip / rotation"]
    end

    subgraph API["外部接口"]
        direction TB
        Ctor["构造函数<br/>初始化路径与增强配置"]
        Get["get(index)<br/>返回 LR/HR 样本"]
        Meta["size() / empty()<br/>get_filename()"]
    end

    subgraph P["内部处理函数"]
        direction TB
        Scan["getImageFiles()<br/>扫描并排序"]
        Load["loadImage()<br/>读取与颜色转换"]
        Aug["applyAugmentation()<br/>同步增强"]
        Tensor["matToTensor()<br/>归一化与 CHW 转换"]
    end

    Ctor --> Scan --> Files
    Files --> Get
    LRDir --> Get
    Size --> Get
    AugCfg --> Aug
    Get --> Load --> Aug --> Tensor
    Get --> Meta
```

### 2.2 完整说明版

```mermaid
classDiagram
    class FaceSRDataset {
        -vector~fs::path~ hr_images_
        -string lr_dir_
        -int hr_size_
        -int lr_size_
        -AugmentationConfig aug_config_
        -bool augment_
        +FaceSRDataset(hr_dir, lr_dir, hr_size, lr_size, augment)
        +FaceSRDataset(hr_dir, lr_dir, hr_size, lr_size, aug_config)
        +get(size_t index) Example
        +size() optional~size_t~
        +get_filename(size_t index) string
        +empty() bool
        +setAugmentationConfig(config)
        -getImageFiles(dir) vector~path~
        -isSupportedImageFormat(extension) bool
        -loadImage(path) cv::Mat
        -applyAugmentation(hr, lr) pair~Mat,Mat~
        -matToTensor(img) Tensor
    }

    class AugmentationConfig {
        +bool enable_flip
        +bool enable_rotation
        +double flip_probability
        +double rotation_probability
    }

    class DataLoaderConfig {
        +int batch_size
        +int num_workers
        +bool shuffle
        +bool drop_last
    }

    FaceSRDataset --> AugmentationConfig
    DataLoaderConfig ..> FaceSRDataset : DataLoaderOptions
```

## 3. 单样本处理流程

### 3.1 论文紧凑版

```mermaid
flowchart LR
    subgraph A["输入索引"]
        direction TB
        Index["DataLoader<br/>get(index)"]
        HRPath["hr_images_[index]"]
    end

    subgraph B["HR/LR 构造"]
        direction TB
        ReadHR["读取 HR<br/>BGR -> RGB"]
        HRResize["HR resize<br/>256 x 256"]
        Choice{"LR 来源"}
        ReadLR["读取同名 LR<br/>64 x 64"]
        OnlineLR["HR 下采样生成 LR<br/>64 x 64"]
    end

    subgraph C["增强与转换"]
        direction TB
        Aug["同步增强<br/>flip / rotate"]
        Float["float32 归一化<br/>[0,1]"]
        CHW["HWC -> CHW"]
    end

    subgraph D["输出样本"]
        direction TB
        LR["LR Tensor<br/>[3,64,64]"]
        HR["HR Tensor<br/>[3,256,256]"]
        Example["Example<br/>data=LR, target=HR"]
    end

    Index --> HRPath --> ReadHR --> HRResize --> Choice
    Choice -- LR 文件存在 --> ReadLR
    Choice -- 在线生成 --> OnlineLR
    HRResize --> OnlineLR
    ReadLR --> Aug
    OnlineLR --> Aug
    HRResize --> Aug
    Aug --> Float --> CHW --> LR
    CHW --> HR
    LR --> Example
    HR --> Example
```

### 3.2 完整说明版

```mermaid
sequenceDiagram
    participant DL as DataLoader Worker
    participant DS as FaceSRDataset
    participant FS as filesystem
    participant CV as OpenCV
    participant Torch as LibTorch

    DL->>DS: get(index)
    DS->>FS: hr_images_[index]
    DS->>CV: imread(HR, IMREAD_COLOR)
    CV-->>DS: cv::Mat BGR
    DS->>CV: cvtColor(BGR -> RGB)
    DS->>CV: resize(HR -> hr_size x hr_size)

    alt lr_dir 非空且同名 LR 文件存在
        DS->>CV: imread(LR)
        DS->>CV: cvtColor(BGR -> RGB)
        DS->>CV: resize(LR -> lr_size x lr_size)
    else 未配置 LR 或 LR 文件不存在
        DS->>CV: resize(HR -> LR, INTER_CUBIC)
    end

    opt augment = true
        DS->>CV: 对 HR/LR 同步 flip
        DS->>CV: 对 HR/LR 同步 rotate 90/180/270
    end

    DS->>Torch: from_blob + clone
    DS->>Torch: normalize 到 [0,1]
    DS->>Torch: permute HWC -> CHW
    DS-->>DL: Example(data=LR, target=HR)
```

## 4. 训练与验证数据流

### 4.1 论文紧凑版

```mermaid
flowchart LR
    subgraph T["训练流"]
        direction TB
        TD["训练 Dataset<br/>augment=true"]
        TL["DataLoader<br/>batch + workers"]
        TB["LR/HR Batch"]
        TM["Generator + Discriminator"]
        Loss["Loss<br/>Pixel / Perceptual / GAN"]
    end

    subgraph V["验证流"]
        direction TB
        VD["验证 Dataset<br/>augment=false"]
        VL["DataLoader<br/>batch=1"]
        VB["LR/HR Sample"]
        VM["Generator eval<br/>NoGrad"]
        Metric["PSNR / SSIM<br/>对比图保存"]
    end

    TD --> TL --> TB --> TM --> Loss
    VD --> VL --> VB --> VM --> Metric
```

### 4.2 完整说明版

```mermaid
flowchart LR
    subgraph TrainFlow["训练数据流"]
        TDataset["FaceSRDataset(train_hr_dir, train_lr_dir, augment=true)"]
        TStack["Stack&lt;Example&gt;"]
        TLoader["DataLoader<br/>batch_size=config.batch_size<br/>workers=config.num_workers<br/>enforce_ordering(false)"]
        TBatch["LR / HR batch"]
        TDevice["to(device)"]
        TModel["Generator + Discriminator"]
        TLoss["Pixel / Perceptual / GAN Loss"]
    end

    subgraph ValFlow["验证数据流"]
        VDataset["FaceSRDataset(val_hr_dir, val_lr_dir, augment=false)"]
        VStack["Stack&lt;Example&gt;"]
        VLoader["DataLoader<br/>batch_size=1<br/>workers=1<br/>enforce_ordering(true)"]
        VBatch["LR / HR sample"]
        VModel["Generator eval + NoGrad"]
        VMetric["PSNR / SSIM<br/>create_comparison_image()"]
    end

    TDataset --> TStack --> TLoader --> TBatch --> TDevice --> TModel --> TLoss
    VDataset --> VStack --> VLoader --> VBatch --> VModel --> VMetric
```

## 5. 推理阶段图像处理流程

训练/验证使用 `FaceSRDataset`，推理阶段不使用该数据集类，而是直接通过 `image_utils` 完成格式转换。

### 5.1 论文紧凑版

```mermaid
flowchart LR
    subgraph A["输入与预处理"]
        direction TB
        In["输入图像<br/>BGR uint8 HWC"]
        ToTensor["mat_to_tensor()<br/>RGB float BCHW"]
        Device["to(device)<br/>CPU / CUDA"]
    end

    subgraph B["模型推理"]
        direction TB
        NoGrad["NoGradGuard"]
        Model["TorchScript / RRDBNet"]
        SR["SR Tensor<br/>clamp(0,1)"]
    end

    subgraph C["后处理与输出"]
        direction TB
        ToMat["tensor_to_mat()<br/>BCHW -> HWC"]
        BGR["RGB -> BGR<br/>uint8"]
        Save["cv::imwrite()<br/>保存结果"]
    end

    In --> ToTensor --> Device --> NoGrad --> Model --> SR --> ToMat --> BGR --> Save
```

### 5.2 完整说明版

```mermaid
flowchart TB
    Input["输入图像文件或 cv::Mat<br/>OpenCV BGR / uint8 / HWC"]
    Read["cv::imread()"]
    MatToTensor["utils::mat_to_tensor()<br/>BGR -> RGB<br/>uint8 -> float32<br/>[0,255] -> [0,1]<br/>BHWC -> BCHW"]
    ToDevice["to(device)<br/>CPU / CUDA"]
    NoGrad["torch::NoGradGuard"]
    Forward["TorchScript 或 RRDBNet<br/>forward(LR) -> SR"]
    Clamp["clamp(0,1)"]
    TensorToMat["utils::tensor_to_mat()<br/>BCHW/CHW -> HWC<br/>[0,1] -> uint8<br/>RGB -> BGR"]
    Save["cv::imwrite()<br/>保存 SR 图像"]

    Input --> Read --> MatToTensor --> ToDevice --> NoGrad --> Forward --> Clamp --> TensorToMat --> Save
```

## 6. 模块职责划分

| 模块 | 文件 | 主要职责 |
| --- | --- | --- |
| 数据集封装 | `include/utils/dataset.h`, `src/utils/dataset.cpp` | 扫描 HR 文件、读取图像、生成或读取 LR、同步增强、转换为训练 Tensor |
| 图像格式转换 | `include/utils/image_utils.h`, `src/utils/image_utils.cpp` | OpenCV `cv::Mat` 与 LibTorch `Tensor` 互转，保存 Tensor 图像，生成对比图 |
| 训练数据入口 | `src/trainer.cpp` | 构造训练 DataLoader，得到 `[B,3,64,64]` LR 与 `[B,3,256,256]` HR |
| 验证数据入口 | `src/trainer.cpp` | 构造无增强验证 DataLoader，计算 PSNR/SSIM 并保存可视化结果 |
| 推理数据入口 | `src/inference.cpp` | 直接读取输入图像，预处理为 Tensor，模型前向后转换回 BGR 图像 |

## 7. 关键数据格式

| 阶段 | 数据结构 | 通道顺序 | 维度 | 数值范围 |
| --- | --- | --- | --- | --- |
| OpenCV 读入 | `cv::Mat` | BGR | HWC | `uint8 [0,255]` |
| Dataset 内部处理 | `cv::Mat` | RGB | HWC | `uint8 [0,255]` |
| Dataset 输出单样本 | `torch::Tensor` | RGB | CHW | `float32 [0,1]` |
| DataLoader 输出 batch | `torch::Tensor` | RGB | BCHW | `float32 [0,1]` |
| 推理输出保存前 | `cv::Mat` | BGR | HWC | `uint8 [0,255]` |

## 8. 当前实现特点

- 支持图像格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tiff`。
- HR 默认尺寸为 `256 x 256`，LR 默认尺寸为 `64 x 64`，对应 4 倍超分辨率。
- 当 `train_lr_dir` 或 `val_lr_dir` 为空时，LR 由 HR 通过 `cv::INTER_CUBIC` 在线下采样得到。
- 数据增强只在训练集启用，验证集关闭增强。
- 数据增强会对 HR 和 LR 同步执行，避免输入与标签空间错位。
- OpenCV 与 LibTorch 的边界转换集中在 `matToTensor()` 和 `image_utils` 中，核心转换是颜色空间、数值范围和维度顺序。
