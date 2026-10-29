# FHEViT

FHEViT 是一套研究代码，用于研究胸片视觉语言模型表示的效用、重建泄露和
缩小规模 CKKS 电路。它不是临床产品，也不声称加密后仍能保护解密表示中的
图像隐私。

## 从这里开始

- [中文科研笔记](docs/RESEARCH_NOTES.zh-CN.md)：研究目标、实现架构、现行证据、
  踩坑和开源发布边界。
- [环境与复现说明](docs/ENVIRONMENT.md)：已验证的软件版本与新环境安装步骤。
- [复现指南](docs/REPRODUCTION.md)：源代码验证、数据/模型本地准备和结果复现边界。
- [数据源与本地目录](data/README.zh-CN.md)：数据来源、对应本地路径、准备脚本和发布边界；[English](data/README.md)。
- [模型快照说明](models/README.md)：模型提供方页面与本地目录；模型文件不会随仓库发布。
- [当前科学状态](docs/handoff/STATUS.md)：当前结论的唯一摘要入口。
- [逐文件代码地图](CODE_MAP.md)：所有 Python 模块、用途和公开符号。
- [评测协议](EVALUATION_PROTOCOL.md)：数据划分、评分、bootstrap 与非劣效规则。
- [相关工作](RELATED_WORK.md)：FHE Transformer、医学影像、表示泄露和模型引用。
- [公开发布流程](docs/PUBLISHING.md)：独立 GitHub 仓库发布、暂存区审核和归档引用步骤。

## 实现范围

仓库实现了胸片预处理、视觉表示与 provenance 契约、患者级效用和隐私审计、
PCA/仿射/空间释放原型、TenSEAL 多项式 CKKS 缩小 block，以及可选的
InternVL2/Qwen2-VL adapter。实验脚本覆盖 NIH ChestX-ray14 数据筛选、前缀效用、
重建攻击、像素学生容量和 CKKS 深度/有限性测量。

当前评分主线是 InternVL2-2B 的明文前缀研究。CKKS 实测对象只是一个缩小 block，
不是完整加密 InternViT。较早的交互式刷新实现和 adapter 作为历史科研基线保留，
不能描述成已完成的安全部署。

## 当前证据

开发分析集上，近满秩、零坐标噪声的 PCA 释放通过预先登记的 PA-only 效用规则；
已测试的坐标噪声水平失败。秩 1024 和 1999 的零噪声释放可被当前重建攻击恢复出
胸片结构。深度不超过 5 的像素学生未通过，多项式视觉塔在第 2 个 block 出现
非有限值。保留确认集仍保持关闭。

这些结果不证明隐私保护、临床等效、无损、SOTA 或首创。引用数字前先读
[当前状态](docs/handoff/STATUS.md) 和 [主张边界](docs/handoff/CLAIMS_AUDIT.md)。

## 验证

Windows PowerShell 下从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

当前本地基线是 139 项测试通过。只克隆源代码可以复现安装、构建和测试，不能仅凭
源代码复现论文聚合结果；后者还需要依法取得完全匹配的数据集、队列清单、模型和
检查点版本以及本地特征缓存。数据、模型、逐病例结果、私有账本、缓存和训练权重
默认不进入公开源代码；细节见 `.gitignore`、[数据源说明](data/README.md) 和
[环境说明](docs/ENVIRONMENT.md)。

## 引用与许可

软件元数据见 [CITATION.cff](CITATION.cff)。本项目采用 Apache License 2.0；
数据集和预训练模型仍受各自提供方的许可与使用条款约束。
