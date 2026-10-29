# FHEViT 工作笔记

本文是面向后续科研人员的统一工作笔记。源码级清单以
[CODE_MAP.md](../CODE_MAP.md) 为准，现行科学结论以
[docs/handoff/STATUS.md](handoff/STATUS.md) 为准。含逐病例字段和本机路径的原始
JSON 保留在本地，不随公开源码发布；旧的工作区汇总已从公开材料中移除，现行结论
以状态页和冻结协议为准。

## 1. 研究问题与边界

项目研究两层问题：

1. 胸片视觉表示经过压缩或受限释放后，是否仍能支持肺炎判别；
2. 密文计算、解密和后续表示释放是否会把可重建的胸片信息交给攻击者。

当前主线是 InternVL2-2B 的明文视觉前缀测量。Qwen2-VL-2B 是早期对照。
TenSEAL 只测量缩小的多项式 CKKS block，不代表完整的 InternViT 加密执行。
交互式 CANR/CADN 和旧适配器是保留的研究基线，不应写成已完成的安全部署。

## 2. 当前可复述的结果

- 开发分析集上，前缀 2000、秩 1999、零坐标噪声的无监督 PCA 释放通过
  预注册 PA-only 规则；加入已测坐标噪声后失败。
- 全 token、秩 1024 和秩 1999 的零噪声释放都被当前重建攻击证明具有可见
  胸片结构，不能据此宣称图像隐私。
- 深度不超过 5 的像素学生没有可行格；锁定的多项式视觉塔在第 2 个 block
  出现非有限值，因此没有可用 AUC。
- 598 名保留肺炎患者和最终 holdout 没有被打开；确认集不是开发集通过的替代品。

数字、样本数、种子、阈值和未完成标志以 `STATUS.md` 的经核对汇总为准；原始
逐病例 JSON 只在授权本地环境中复核。历史报告中没有结构化结果支持的峰值或估计
不得写入论文结论。

## 3. 架构分层

```text
胸片/数据清单
      |
      v
preprocessing + NIH cohort rules
      |
      v
FHEViT encoder and anatomical token routing (PyTorch)
      |
      +--> plaintext PCA / affine / spatial release --> InternVL2 or Qwen adapter
      |
      +--> reduced MedFHEBlock (TenSEAL CKKS) --> depth/finite/cosine diagnostics
      |
      +--> privacy_audit / reconstruction / attribute / membership checks
```

代码分组如下；每个文件和公开符号已经逐项列在 `CODE_MAP.md`。

| 分组 | 内容 |
|---|---|
| `fpavlm/` 核心 | 配置、预处理、checkpoint/provenance、评测、隐私审计、release contract |
| `fpavlm/cipher_native/` | CKKS tensor 边界、PolyNorm、低阶激活、token mixer、MedFHEBlock、刷新与 DP 记账 |
| `fpavlm/llm/` | LLM 接口、InternVL2/Qwen2-VL adapter、噪声感知解码 |
| `experiments/` | 数据选择、NIH 主线、重建/学生/多项式/CKKS 实验及结果读取器 |
| `tests/` | 139 个单元/边界/协议回归测试 |
| `tools/` | 项目工作区审计与公开发布检查 |

## 4. 重要实现约束

- 患者级划分优先于图像级划分；PA/AP 选择、年龄范围和标签规则锁在协议中。
- `patient_id`、图像路径、报告文本和训练缓存属于本地数据面；它们不应进入
  公开 JSON、图表或提交历史。
- 加密边界由 `CiphertextTensor` 等类维护；服务端路径不允许解密后再加密的
  fallback。交互式刷新会暴露中间激活，必须单独标注信任模型。
- checkpoint 必须经过 provenance 和 SHA-256 manifest 校验；缺失本地权重时，
  公开源代码仍应能安装并运行不依赖权重的测试。
- 未完成的 AUC、关闭的确认集、失败的深度门和负结果不能被后续摘要抹掉。

## 5. 踩坑与已处理事项

1. **版本漂移**：源码版本以 `pyproject.toml` 与 `fpavlm.__version__` 为准；
   安装后核对 `importlib.metadata.version("fpavlm")`，旧环境可能持有过期元数据。
2. **代码地图过期**：地图由 `experiments/audit_workspace.py --write` 根据 AST 生成，
   源码增删后必须重新生成并执行 `--check`。
3. **结果污染**：逐病例 JSON 可能保留 patient ID、绝对路径或文本；结果目录
   仅作本地证据，公开仓库使用本笔记和去标识化汇总。
4. **数据集交集误读**：NIH 和 COVID 使用不同编号空间，字符串相同不证明同一
   患者；OpenI 使用 `uid`。跨数据集审计必须标注 ID namespace。
5. **数学与安全边界**：PCA 通过不等于隐私，CKKS cosine 接近不等于临床等效，
   DP 记账不等于重建攻击失败。多项式近似的有限性必须先于 AUC。
6. **路径泄露**：OpenI 本地元数据曾写入本机绝对图像路径；这类数据由
   `.gitignore` 和 `tools/public_release_check.py` 阻止进入源发布。
7. **审计脚本副作用**：审计脚本只允许生成索引或检查，不删除数据、不打开
   holdout、不刷新私有 ledger。

## 6. 公开发布清单

- [ ] `pytest` 全部通过，`audit_workspace --check` 通过；
- [ ] `tools/public_release_check.py` 通过；
- [ ] Git 暂存区不包含 `data/`、`models/`、`artifacts/`、`private-client/`、
  `experiments/results/`、`.venv/` 或 `docs/handoff/work/`；
- [ ] 没有本机绝对路径、访问令牌、私钥、病例 ID、报告文本或未授权模型文件；
- [ ] README、环境说明、许可证、CITATION.cff 和相关工作链接齐全；
- [ ] 所有数字都能回链到协议和结果 JSON，未把开发集结论写成确认集结论。

## 7. 引用起点

项目应同时引用数据集、模型和加密库的原始工作，而不是只引用本仓库：

- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*,
  CVPR 2017（NIH ChestX-ray14 的数据基础）；
- OpenGVLab, *InternVL 2.0: Scaling Up Multimodal Large Language Models*, 2024；
- Alibaba, *Qwen2-VL Technical Report*, 2024（仅在 Qwen 对照被使用时）；
- Benaissa et al., *TenSEAL: A Library for Encrypted Tensor Operations Using
  Homomorphic Encryption*, 2021；
- 相关 FHE Transformer、split-inference inversion、医学隐私和 DP 工作见
  [RELATED_WORK.md](../RELATED_WORK.md)。

引用这些工作是为了说明数据、模型和方法来源，不构成“首创”或临床可用声明。
