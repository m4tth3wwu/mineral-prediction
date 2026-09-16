# PPP 岩性与空间验证改进说明

## 改进内容

新增分析入口：`PPP_lithology_spatial_validation.py`。原有 ACAWLR 脚本和结果均未覆盖。

主要变化：

1. 因变量从“矿床周围 2 km 缓冲内的正网格”改为规则积分网格内的矿床事件数。
2. 44 个训练矿床各自只贡献一个事件；16 个验证矿床始终封存。
3. 全研究区等面积规则网格用于近似 Poisson 点过程似然中的空间积分，不再把背景解释为确认无矿。
4. 从 USGS GeMS `MapUnitPolys.shp` 的 `Symbol` 属性读取岩性；`Igneous, intrusive` 为侵入岩，不使用 PDF 像素颜色。
5. 比较三组模型：无岩性、侵入岩 0/1、侵入岩 0/1 加距侵入岩接触带距离。
6. 验证采用 200 km 空间块、0/25/50 km 死区、折内成对 AUC，以及 50 km 死区下的抽样 LPO-SCV 风格审计。
7. 最终输出是相对矿床事件强度，不是绝对成矿概率。

## 推荐运行

当前推荐 10 km 积分网格：

```powershell
D:\anaconda\python.exe .\PPP_lithology_spatial_validation.py `
  --grid-km 10 `
  --output-dir .\output\ppp_lithology_spatial_validation_10km
```

第一次运行如需强制重算特征，追加：

```powershell
--recompute-features
```

快速检查：

```powershell
D:\anaconda\python.exe .\PPP_lithology_spatial_validation.py --smoke
```

## 已完成的尺度敏感性

下表均为包含“侵入岩 0/1 + 距侵入岩接触带距离”的 M2 模型。

| 积分网格 | 积分单元数 | 50 km 死区折内成对 AUC | 16 点 presence-background AUC | 16 点 Top 5% 命中 | 16 点 Top 10% 命中 |
|---:|---:|---:|---:|---:|---:|
| 20 km | 8,305 | 0.851 | 0.759 | 8/16 | 8/16 |
| 10 km | 33,220 | 0.914 | 0.924 | 11/16 | 12/16 |
| 5 km | 132,888 | 0.942 | 0.924 | 11/16 | 12/16 |

10 km 与 5 km 的外部排序和矿床命中已经接近，因此 10 km 是当前计算量与稳定性的合理折中。两种分辨率的全区 log 强度相关系数约为 0.948，但 Top 靶区边界仍有差异，不能宣称空间靶区已经完全收敛。

推荐的 10 km 结果还完成了 50 km 死区、88 个正背景对的抽样 LPO-SCV 风格审计，成对排序 AUC 为 0.920。

## 20 km 正式审计结果

- 无岩性模型在 50 km 死区下的折内成对 AUC：0.714。
- 仅加入侵入岩 0/1：0.733。
- 再加入距侵入岩接触带距离：0.851。
- M2 的 50 km 抽样 LPO-SCV 风格成对 AUC：0.826，共 86 个正背景对。
- 20 km 较粗，主要用于方法审计，不作为最终推荐制图尺度。

## 解释注意事项

- AUC 中的“负类”是研究区背景，不是地质学确认无矿区，因此指标只表示矿床相对背景的排序能力。
- 接触带距离带来的增益明显高于单纯侵入岩 0/1，说明矿床与岩体邻近关系比“是否正好落在粉色多边形内”更有信息。
- M2 中同时加入接触带距离后，侵入岩 0/1 系数可能缩小甚至变号。这是两个相关变量的条件系数，不应单独解释成“侵入岩不利成矿”。
- USGS 图件比例尺为 1:1,000,000。5 km 或 10 km 网格只是积分和预测分辨率，不代表原始岩性边界具有同等精度。
- 16 个矿床的 Top 5%/10% 命中数是主外部结果；外部 AUC仅作排序补充。

## 主要输出

- `spatial_cv_summary.csv`：不同模型与死区半径的空间验证汇总。
- `spatial_fold_metrics.csv`：逐折结果和死区删除数量。
- `sampled_lpo_scv_pairs.csv`：由同一重拟合模型比较的正背景对。
- `external_16_deposit_scores.csv`：16 个封存矿床的强度、全区百分位和 Top 面积命中。
- `ppp_coefficients.csv`：标准化连续变量及 0/1 岩性变量的系数和强度比。
- `quadrature_predictions.csv`：全研究区积分网格与三种模型的相对强度。
- `validation_dashboard.png`：空间迁移、面积效率、外部命中和岩性系数图。
