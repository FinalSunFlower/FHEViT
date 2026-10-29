# 该改哪里

先读 `README.md`、`docs/RESEARCH_NOTES.zh-CN.md` 和 `STATUS.md`。结果目录和本地
数据不是公开源代码的一部分；修改结果前必须先确认协议是否允许新增运行。

## 现行 NIH 主线

| 要改的事 | 文件 |
|---|---|
| 病人、视图、年龄、标签规则 | `experiments/nih_pneumonia.py` |
| 无标签池选片 | `experiments/nih_unlabeled_pool.py` |
| 通过格的桥 | `experiments/validate_nih_unlabeled_bridge.py` |
| 噪声前沿 | `experiments/validate_nih_coordinate_frontier.py` |
| 0.05 与 0.1 的预算 | `experiments/validate_nih_fine_noise.py` |
| 重建与属性 | `experiments/validate_nih_passing_release_attack.py` |
| 像素学生 | `experiments/validate_nih_pixel_students.py` |
| 像素 R² | `experiments/validate_nih_pixel_r2.py` |
| 多项式塔 | `experiments/validate_nih_poly_tower.py` |
| 已测的多项式块 | `fpavlm/cipher_native/med_fhe_vit_block.py` |
| 锁定规则 | `EVALUATION_PROTOCOL.md`；新增规则只能追加 |
| 结果数字 | 本地 `experiments/results/*.json`；公开文档只引用已审查的聚合字段 |

这些脚本已经跑完。没有新的锁定规则之前，不要改系数、秩、岭惩罚、margin 或 PCA 截止后再评分。

## 仍在树里、但不是现行 NIH 结论

`fpavlm/pipeline.py` 和 `fpavlm/cipher_native/canr.py` 是更早的交互式刷新实现。客户端看得到中间激活。`CODE_MAP.md` 里出现的模块是还留着的文件，不是第二套现行结论。

`experiments/audit_workspace.py` 是源码地图和本地资产哈希检查器；
`tools/public_release_check.py` 是源发布前的敏感路径/凭证扫描器。两者都不删除
数据、不打开 holdout，也不修改实验结果。
