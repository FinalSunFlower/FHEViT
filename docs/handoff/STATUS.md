# 当前状态

本页是唯一的现行科学摘要。评测规则及后续修订见 `EVALUATION_PROTOCOL.md`，
公开主张边界见 `CLAIMS_AUDIT.md`。逐病例结果 JSON 是本地证据，不随公开源码
发布；本页只记录经核对的聚合结论。

## 项目在做什么

FHEViT 要回答两件事：减少后的视觉前缀还能不能做肺炎判别，以及解密这些坐标之后，胸片会不会被还原。现行主线是 InternVL2-2B 的明文 token。Qwen2-VL-2B 只出现在更早的对照里。0.4 有限状态符号的继续测试已经停止。确认集保持关闭。

## 现行结论

开发集上，前缀 2000、秩 1999、零噪声的无监督 PCA 释放，按预先写下的 PA-only 规则通过。同一类释放在坐标噪声达到训练标准差的 0.25 倍时不再通过。细网格把两个秩的误差预算记为 0.1 倍训练标准差。这不是差分隐私。

对全 token、秩 1024、秩 1999 的零噪声释放，视图条件均值的 SSIM 优势下界都大于 0。对这一名攻击者，解密释放会还原胸片。深度至多 5 的两条像素学生都是 0/5。锁定的多项式视觉塔在第 2 个块失去有限值，没有 PA 分数。保留的 598 名肺炎患者没有下载、没有评分。

因此：开发集上的效用是明文近满秩释放的效用；当前近似进不了可用电路；若将来近似成功，解密仍会把图像交给这名攻击者。

## 通过与失败

| 条目 | 字段 | 值 | 文件 |
|---|---|---|---|
| PA-only 通过格 | `pa_only.noninferior` | true | `nih_unlabeled_bridge_internvl2-2b.json` |
| 该格 AUC 差 | `pa_only.difference` | −0.005400408679575808 | 同上，`prefix` 2000，`rank_order` 1999 |
| 该格单侧下界 | `pa_only.one_sided_95_lower` | −0.03301969271417782 | 同上 |
| 零噪声秩 1024 | `refit_passes` | 5 / 5 | `nih_coordinate_frontier_internvl2-2b.json` |
| 零噪声秩 1999 | `refit_passes` | 4 / 5 | 同上 |
| 尺度 0.25 及以上 | 各格 `passes` | false | 同上 |
| 误差预算 | `error_budget` | 1024: 0.1，1999: 0.1 | `nih_fine_noise_internvl2-2b.json` |
| 全 token 视图 SSIM 下界 | `versus_view_conditional_mean.ci95_lower` | 0.21528252576431173 | `nih_passing_release_attack_internvl2-2b.json` |
| 秩 1024 视图 SSIM 下界 | 同上 | 0.1787551244970762 | 同上 |
| 秩 1999 视图 SSIM 下界 | 同上 | 0.2046160706013421 | 同上 |
| 像素学生 | `no_feasible_student` | true | `nih_pixel_students_internvl2-2b.json` |
| 多项式塔 | `first_nonfinite_block` | 2 | `nih_poly_tower_internvl2-2b.json` |
| 多项式塔效用 | `auc_computed` | false | 同上 |
| 保留确认 | `images_downloaded` | false | `data/nih_chestxray14_final_holdout/positives.json` |
| 视图捷径 | `view_shortcut_supported` | true | `nih_sealed_strata.json` |
| 前缀网格 | `grid_failed` | true | `nih_prefix_grid_internvl2-2b.json` |

主终点是 PA-only。全集合上的通过不代替这条规则。AP 只作探索。

## 越界表述

加密后无损、保护了图像隐私、首创、把开发集上的通过写成已经完成的非劣效释放方案，均属越界表述。数值异常应只按记录的有限性标志描述，不引用未经 JSON 支持的峰值。

## 还没做

没有打开保留确认集；没有完成完整 FHE vision tower、带自举运行时的端到端测量，
也没有外部临床验证。未来新实验必须先锁定协议与数据访问边界。

允许的对外句子在 `CLAIMS_AUDIT.md`。
