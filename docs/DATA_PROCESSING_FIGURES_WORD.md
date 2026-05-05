# 数据处理模块论文插图版

本文件只放适合 Word/论文正文使用的 Mermaid 图。图片参数已按《西安石油大学本科毕业设计（论文）模板（2024版）》中的图格式要求调整：节点文字短、图形接近 `4:3`、适合导出后按 `9.00 cm x 6.75 cm` 插入正文。

导出建议：在 Mermaid Live 中使用 `Actions -> Download SVG`，再插入 Word。SVG 放大不模糊。

Word 中设置建议：

- 图片版式：`嵌入型` 或 `上下型`，不要使用 `浮于文字上方`。
- 图片位置：居中。
- 图片尺寸：常规使用 `宽 9.00 cm，高 6.75 cm`；内容较多时可用 `宽 13.50 cm，高 9.00 cm`。
- 图题位置：图片正下方。
- 图题格式：宋体，五号，加粗，居中；图号按章编号，例如 `图5.1  数据处理模块总体架构`。
- 图与上文留一行空格，图题与下文留一行空格。
- 同类图片尺寸尽量统一。

尺寸建议：

| 图号 | 建议尺寸 | 说明 |
| --- | --- | --- |
| 图5.1 | `13.50 cm x 9.00 cm` | 总体架构内容较多，建议用模板允许的较大尺寸 |
| 图5.2 | `9.00 cm x 6.75 cm` | 内部结构图，常规尺寸即可 |
| 图5.3 | `9.00 cm x 6.75 cm` | 单样本流程，已改为三段式 |
| 图5.4 | `9.00 cm x 6.75 cm` | 训练/验证双流程，对称结构 |
| 图5.5 | `9.00 cm x 6.75 cm` | 推理流程，已改为三段式 |

## 图 5.1 数据处理模块总体架构

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "fontFamily": "SimSun, Microsoft YaHei, Times New Roman", "primaryColor": "#FFFFFF", "primaryTextColor": "#000000", "primaryBorderColor": "#000000", "lineColor": "#000000", "clusterBkg": "#FFFFFF", "clusterBorder": "#000000"}}}%%
flowchart TB
    subgraph Row1[" "]
        direction LR
        A["配置层<br/>TrainConfig<br/>尺寸/路径/batch"]
        B["输入层<br/>HR目录<br/>可选LR目录"]
        C["文件发现<br/>扫描/过滤/排序"]
    end

    subgraph Row2[" "]
        direction LR
        D["样本构造<br/>读取HR<br/>生成或读取LR"]
        E["预处理增强<br/>resize<br/>flip/rotate"]
        F["张量转换<br/>归一化<br/>HWC->CHW"]
    end

    subgraph Row3[" "]
        direction LR
        G["批处理<br/>Stack<br/>DataLoader"]
        H["训练/验证<br/>LR/HR batch"]
        I["模型消费<br/>Generator<br/>Loss/Metric"]
    end

    A --> B --> C
    C --> D
    D --> E --> F
    F --> G
    G --> H --> I
```

## 图 5.2 FaceSRDataset 内部结构

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "fontFamily": "SimSun, Microsoft YaHei, Times New Roman", "primaryColor": "#FFFFFF", "primaryTextColor": "#000000", "primaryBorderColor": "#000000", "lineColor": "#000000", "clusterBkg": "#FFFFFF", "clusterBorder": "#000000"}}}%%
flowchart TB
    subgraph State["内部状态"]
        direction LR
        S1["hr_images_<br/>HR文件列表"]
        S2["lr_dir_<br/>LR路径"]
        S3["hr/lr size<br/>256/64"]
        S4["aug_config_<br/>增强参数"]
    end

    subgraph API["外部接口"]
        direction LR
        A1["构造函数"]
        A2["get(index)"]
        A3["size/empty<br/>filename"]
    end

    subgraph Func["内部函数"]
        direction LR
        F1["getImageFiles<br/>扫描排序"]
        F2["loadImage<br/>读取转RGB"]
        F3["applyAug<br/>同步增强"]
        F4["matToTensor<br/>转Tensor"]
    end

    A1 --> F1 --> S1
    S1 --> A2
    S2 --> A2
    S3 --> A2
    S4 --> F3
    A2 --> F2 --> F3 --> F4
    A2 --> A3
```

## 图 5.3 单样本处理流程

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "fontFamily": "SimSun, Microsoft YaHei, Times New Roman", "primaryColor": "#FFFFFF", "primaryTextColor": "#000000", "primaryBorderColor": "#000000", "lineColor": "#000000", "clusterBkg": "#FFFFFF", "clusterBorder": "#000000"}}}%%
flowchart TB
    subgraph R1[" "]
        direction LR
        A["DataLoader<br/>请求 index"]
        B["读取 HR<br/>BGR->RGB"]
        C["HR resize<br/>256x256"]
    end

    subgraph R2[" "]
        direction LR
        D{"LR 来源"}
        E["读取 LR<br/>64x64"]
        F["HR下采样<br/>Bicubic 64x64"]
    end

    subgraph R3[" "]
        direction LR
        G["同步增强<br/>flip / rotate"]
        H["归一化<br/>float32 [0,1]"]
        I["维度转换<br/>HWC->CHW"]
    end

    J["输出 Example<br/>data=LR<br/>target=HR"]

    A --> B --> C --> D
    D -- "有LR文件" --> E
    D -- "在线生成" --> F
    E --> G
    F --> G
    C --> G
    G --> H --> I --> J
```

## 图 5.4 训练与验证数据流

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "fontFamily": "SimSun, Microsoft YaHei, Times New Roman", "primaryColor": "#FFFFFF", "primaryTextColor": "#000000", "primaryBorderColor": "#000000", "lineColor": "#000000", "clusterBkg": "#FFFFFF", "clusterBorder": "#000000"}}}%%
flowchart TB
    subgraph Train["训练流"]
        direction LR
        T1["训练Dataset<br/>augment=true"]
        T2["DataLoader<br/>batch/workers"]
        T3["LR/HR Batch"]
        T4["G/D训练"]
        T5["组合损失"]
    end

    subgraph Val["验证流"]
        direction LR
        V1["验证Dataset<br/>augment=false"]
        V2["DataLoader<br/>batch=1"]
        V3["LR/HR Sample"]
        V4["Generator eval"]
        V5["PSNR/SSIM<br/>对比图"]
    end

    T1 --> T2 --> T3 --> T4 --> T5
    V1 --> V2 --> V3 --> V4 --> V5
```

## 图 5.5 推理阶段图像处理流程

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "fontFamily": "SimSun, Microsoft YaHei, Times New Roman", "primaryColor": "#FFFFFF", "primaryTextColor": "#000000", "primaryBorderColor": "#000000", "lineColor": "#000000", "clusterBkg": "#FFFFFF", "clusterBorder": "#000000"}}}%%
flowchart TB
    subgraph P1["预处理"]
        direction LR
        A["输入图像<br/>BGR uint8"]
        B["mat_to_tensor<br/>RGB float BCHW"]
        C["to(device)<br/>CPU/CUDA"]
    end

    subgraph P2["模型推理"]
        direction LR
        D["NoGradGuard"]
        E["TorchScript<br/>或 RRDBNet"]
        F["SR Tensor<br/>clamp(0,1)"]
    end

    subgraph P3["后处理"]
        direction LR
        G["tensor_to_mat<br/>BCHW->HWC"]
        H["RGB->BGR<br/>uint8"]
        I["cv::imwrite<br/>保存结果"]
    end

    A --> B --> C --> D --> E --> F --> G --> H --> I
```
