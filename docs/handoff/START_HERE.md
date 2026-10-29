# 新对话从这里读

当前公开入口是 `README.md` 与 `docs/RESEARCH_NOTES.zh-CN.md`。本地的逐病例 JSON
不得上传；此处只保留经过审查的汇总结论。若汇总结论与评测协议冲突，先核对协议。

| 问题 | 答案 | 出处 |
|---|---|---|
| 项目是什么 | 研究减少后的胸片视觉前缀还能不能做肺炎判别，以及解密后胸片会不会被还原。主线是 InternVL2-2B 明文 token | `STATUS.md` |
| 最重要的现行结论 | 开发集上近满秩零噪声 PCA 通过 PA-only；加噪后失败。零噪声秩 1024 和 1999 能被还原成胸片。深度 ≤5 没有可用学生。多项式塔在第 2 块溢出。确认集未打开 | `STATUS.md` |
| 什么通过、什么失败 | 通过：前缀 2000、秩 1999、零噪声。失败：前缀网格、尺度 ≥0.25、像素学生 0/5、多项式塔无 AUC | `STATUS.md` |
| 哪些名单封着、哪些看过 | 保留确认集保持关闭；细节遵从评测协议，不公开名单或哈希 | `EVALUATION_PROTOCOL.md` |
| 通过规则 | PA-only 上 AUC 差的单侧 95% 下界大于 −0.05；五次里至少四次。bootstrap 2000，种子 42 | `CONVENTIONS.md` |
| 为什么不能写保护了图像隐私 | 秩 1024 与 1999 的视图条件 SSIM 优势下界都大于 0。解密把图像交给这名攻击者 | `CLAIMS_AUDIT.md`，`STATUS.md` |
| FHE 卡在哪里 | TenSEAL 不自举。已测块用掉输出深度 5，预算 6。像素学生深度 1 和 2 都是 0/5 | `ARCHITECTURE_CURRENT.md` |
| 多项式塔为什么失败 | 锁定替换后 `tokens_finite` false，`first_nonfinite_block` 2，`auc_computed` false | `STATUS.md` |
| 为什么没开确认集 | 噪声、学生和近似都还不是一份固定释放；保留确认集的 AUC 未计算 | `STATUS.md` |
| 怎样复核一个格 | 使用公开协议和实验脚本；原始逐病例结果只在授权本地环境中复核 | `EVALUATION_PROTOCOL.md`，`CODE_MAP.md` |
| 下一步 | 先保持确认集关闭；任何新释放、攻击或运行时先追加协议并记录版本 | `EVALUATION_PROTOCOL.md` |
| 哪些主张越界 | 加密后无损；保护了图像隐私；首创；把开发集通过写成已完成的非劣效方案 | `CLAIMS_AUDIT.md` |

改代码前读 `WHERE_TO_EDIT.md`。`STATUS.md` 只保留当前科学摘要；长篇历史实验日志保留在本地，不随源发布。
