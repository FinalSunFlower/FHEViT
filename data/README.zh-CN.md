# 数据来源与本地目录

本索引对应的英文版为 [README.md](README.md)。公开源码不包含影像、患者清单、
队列清单、特征缓存或逐病例结果。`.gitignore` 排除 `data/` 下除本索引外的内容；
新克隆后需要按下表从原始提供方依法申请或获取数据，并仅保存在本地。
队列 manifest 等派生元数据也可能含患者 ID、标签、图像路径、日期或哈希，按敏感数据处理。

## 数据目录索引

| 数据集 / 用途 | 来源与引用 | 本地目录 | 代码入口与状态 |
|---|---|---|---|
| NIH ChestX-ray14 当前探针研究 | [NIH Clinical Center 数据页](https://nihcc.app.box.com/v/ChestXray-NIHCC)；引用 Wang 等人的 [ChestX-ray8 论文](https://arxiv.org/abs/1705.02315) | `data/nih_chestxray14/Data_Entry_2017_v2020.csv`、`data/nih_chestxray14/images/`、本地 `metadata.json` | `experiments/fetch_nih_probe.py`，之后按 `EVALUATION_PROTOCOL.md` 执行冻结步骤 |
| NIH 开发与分析队列 | 同上 | `data/nih_chestxray14_replication/`、`data/nih_chestxray14_confirmation/`、`data/nih_chestxray14_unlabeled_pool/`；生成的清单与影像均留在本地 | `experiments/fetch_nih_replication.py`、`experiments/nih_unlabeled_pool.py`、`experiments/fetch_nih_unlabeled_pool.py`；需按评测协议顺序执行 |
| NIH 封存确认集与最终留出集 | 同上；受评测协议保留 | `data/nih_chestxray14_sealed_confirmation/`、`data/nih_chestxray14_final_holdout/` | 保留输入。除非 `docs/handoff/STATUS.md` 中的协议状态经正式更新，否则不要下载、打开、重建或评分 |
| OpenI / Indiana University 胸片 | [美国国家医学图书馆 Open-i](https://openi.nlm.nih.gov/)；在其搜索/下载界面查找公开胸片与报告 | `data/openi_subset/metadata.json`、`data/openi_subset/images/`；metadata 记录需含 `image_filename` 字段 | 由 `experiments/common.py` 等读取；仅用于旧版 quickstart / adapter 示例，不属于当前评分研究。本仓库不提供 OpenI 批量下载器或子集 manifest |
| COVID-19 胸片队列原型 | [IEEE 8023 COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)；脚本固定源代码提交并保留逐图许可信息 | `data/covid_cxr2_source_metadata.csv`、`data/covid_cxr2_privacy_{development,confirmation,final_holdout}/` | `experiments/download_patient_cohort.py`；历史原型，不是当前 NIH 结果的输入 |

## NIH 下载说明

NIH 数据体量较大，获取时须遵循提供方访问条款。当前探针脚本读取
`Data_Entry_2017_v2020.csv`，按预先固定的筛选规则选择记录，再从
`experiments/nih_pneumonia.py` 引用的第三方 Hugging Face 镜像按需下载 PNG。
CSV URL 固定到一个提交，但影像 ZIP 使用镜像的 `main` 分支，**不是不可变归档版本**；
镜像也不应被描述成 NIH 官方主机。严格复现时应从 NIH 合法获取原始数据，并在本地
记录归档及 metadata 的 SHA-256。

准备初始探针数据的命令只会写入被忽略的 `data/`：

```powershell
.\.venv\Scripts\python.exe experiments\fetch_nih_probe.py
```

之后只能依照 `EVALUATION_PROTOCOL.md` 中冻结的顺序与排除规则继续。最终确认集保持封存；
不要为了让目录看起来完整而运行封存集下载器。缺少保留集是协议边界，不是仓库缺文件。

## 新克隆与复现边界

新克隆后，以上提供方链接用于找到对应数据或申请访问；仓库不会镜像或自动下载所有数据。
OpenI quickstart 需要从 Open-i 获取合法公开数据并整理成表中布局，本仓库没有自动准备器。
每个本地队列应在私有 manifest 中记录来源、版本/提交、访问条款和 SHA-256；若 manifest 含
患者 ID、标签、影像路径或其他受限元数据，不得发布。

只克隆源码即可安装包并运行合成/单元测试；复现科学聚合结果还需要合法取得完全匹配的数据、
本地队列清单、模型/检查点版本和特征缓存。不要用替代数据冒充复现，也不要打开封存确认集。
详见 [复现指南](../docs/REPRODUCTION.md)、[环境说明](../docs/ENVIRONMENT.md)、
[评测协议](../EVALUATION_PROTOCOL.md) 与 [当前科研状态](../docs/handoff/STATUS.md)。
