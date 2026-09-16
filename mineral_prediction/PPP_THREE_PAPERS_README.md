# 三篇论文的简化改进版

入口：`PPP_three_papers_simple.py`。保留原版 ACAWLR 和 PPP 脚本，不覆盖历史结果。

## 三项主要改动

1. **抽样偏差：先做敏感性对照。** 使用已有的“距最近地化采样点距离”构建一个覆盖代理，M3 = M2 + log1p(距离)。记录训练点、外部点及研究区的距离分布。不把背景当真无矿点，也不声称这个代理等于真实勘查努力。它可能同时关联地形、地质或采样设计。
2. **点过程：矿点位置不再挪动。** 新的目标函数为 `sum_events(log(lambda)) - sum_quadrature(area * lambda) - alpha/2 * sum(slopes**2)`。矿点项用真实坐标处的特征；积分项使用面积权重。未惩罚的截距解析求解，使训练区积分总量等于训练矿点数。这个总量是拟合资料的数量，不是未知矿床总数。
3. **空间验证：重复划分并检查隔离。** 200 km 空间块、50 km 隔离、五折、三组固定种子。矿点按真实位置分块，计算到测试方块的实际距离，排除隔离带中的训练点和积分点。每折重新估计填补和标准化参数；正则化固定为 0.1，不根据测试结果调参。

## 控制复杂度的选择

- 不增加神经网络、空间随机场、地形或三维资料。
- 复用原版 5 km 特征缓存，不重复读取全部原始地理数据。
- 四个对照模型：M0 原基础变量；M1 增加侵入岩二分类；M2 再增加侵入岩图斑边界距离；M3 再增加采样覆盖代理。
- 10 km 积分由固定的 5 km 陆地网格聚合：加总面积，每组选择一个真实的细网格点作为代表，不平均岩性类别。5 km / 10 km 近似总面积相同。
- 16 个外部矿点只在全部内部试验后评价，无自动择优过程；它们在此前研究里已被使用，不是新的盲测。
- 区域中预先可用的地化插值、岩性和重力资料视为固定协变量；不能据此宣称能预测完全没有调查资料的新地区。

## 本次实际结果

最终结果目录：`output/ppp_three_papers_simple_v2/`。

|模型|三次空间验证平均 AUC|最高分前10%区域外部命中|
|---|---:|---:|
|M0 不加岩性|0.863|10/16|
|M1 加侵入岩类别|0.893|9/16|
|M2 再加图斑边界距离|0.949|12/16|
|M3 再加覆盖代理|0.946|11/16|

本轮 M2 的结果更一致；不建议把 M3 宣称为成功的偏差校正。表格仅比较本轮同一流程的四个模型，不直接与旧版网格计数 AUC 比高低。没有重新训练原版 ACAWLR。

M2 的三次空间 AUC 范围为 0.940–0.953。这是划分敏感性范围，不是置信区间。5 km 积分在相同第一组划分下的 AUC 为 0.942；全量拟合在两种积分设置下的排序相关约 0.9997，前10%区域面积交并比约 0.942。它说明对本次 10 km → 5 km 加密不太敏感，不证明更细尺度或真实海岸面积已经收敛。

## 文件与运行

需要 `D:\anaconda\python.exe` 中现有的 numpy、pandas、scipy、scikit-learn、pyproj、matplotlib、threadpoolctl；保留同目录的原版 PPP 模块。

默认读取 `output/ppp_lithology_spatial_validation_5km/` 下的三个特征表和 `config.json`。输入文件与代码均记录 SHA-256，便于核对缓存来源；本次没有重新生成原始特征。

重新运行时指定一个**新的输出目录**，程序不会覆盖已有结果：

```powershell
& 'D:\anaconda\python.exe' '.\PPP_three_papers_simple.py' --fine-cv --output-dir '.\output\ppp_three_papers_rerun'
& 'D:\anaconda\python.exe' -m unittest -v test_ppp_three_papers_simple
```

主要结果文件：

- `README_results.md`：完整表格、设置和限制。
- `spatial_cv_summary.csv`、`spatial_repeat_metrics.csv`：重复空间验证。
- `external_validation_summary.csv`：16 点的探索性外部比较。
- `quadrature_convergence.csv`：积分精度检查。
- `sampling_coverage_audit.csv`：采样覆盖差异。
- `boundary_quadrature_audit.csv`：网格边界缺口中的矿点。
- `predictions_5km.csv`：四个模型的全区输出，以及 M3 固定覆盖代理后的相对强度。
- `fitted_models.json`：系数、填补及标准化参数。

## 仍保留的近似

陆地区域及隔离带面积以网格中心近似，不是精确多边形裁切。Kelsey 在地质图陆地内，但所在小格被旧版中心筛选遗漏，最近积分点约 3.22 km；新版没有挪动或删除这个矿点，单独报告该边界误差。44 个点来自论文匹配子集，其发现和选入概率仍未知。岩性边界仍为旧版图斑边界，可能包括内部制图接缝。

## 文献依据

- [Phillips et al. 2009：抽样偏差](https://doi.org/10.1890/07-2153.1)
- [Renner et al. 2015：点过程与积分](https://doi.org/10.1111/2041-210X.12352)
- [Roberts et al. 2017：结构化交叉验证](https://doi.org/10.1111/ecog.02881)
