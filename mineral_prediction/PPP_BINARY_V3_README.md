# 当前推荐入口：粉色二分类 PPP

主程序：`PPP_binary_lithology_v3.py`。

## 执行

在科研训练目录运行：

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda\python.exe' .\PPP_binary_lithology_v3.py --output-dir .\output\ppp_binary_lithology_rerun
```

程序要求新的结果目录，保留历史结果。缓存通过原始数据、代码、参数及缓存文件本身的 SHA-256 校验；一致时复用，否则重新生成。

```powershell
& 'D:\anaconda\python.exe' -m unittest -v test_ppp_binary_lithology_v3 test_ppp_three_papers_simple
```

## 数据与代码位置

- `D:/code/ResearchPractice/data/geochemistry`：地化。
- `D:/code/ResearchPractice/data/faults`：断层全套文件。
- `D:/code/ResearchPractice/data/gravity`：完整重力包。
- `D:/code/ResearchPractice/data/lithology/NGMDB_GeMS_4742`：完整岩性包。
- `D:/code/ResearchPractice/scripts`：原 ACAWLR 脚本及当前 PPP 启动入口。
- `D:/code/ResearchPractice/outputs`：迁移后的 ACAWLR 历史结果。
- 本目录 `output/paper_point_matching`：论文重建44/16矿点。
- 本目录 `output/ppp_lithology_spatial_validation_5km`：固定旧网格位置与面积支撑。
- 本目录 `output/ppp_binary_features_v3`：从原始资料重算的新特征与完整校验清单。
- 本目录 `output/ppp_binary_lithology_v3`：本轮新结果。

目录迁移清单位于 `D:/code/ResearchPractice/organization_manifest.csv`。
可执行代码中的路径已同步修改，历史结果的日志与配置保持原始记录。

## 本轮改进

1. 根据源图例字段 `Symbol == 'Igneous, intrusive'` 定义粉色为1，其余有效岩性为0。
2. 合并同类图斑，使用粉色与其他有效岩性的共同边界；不将同类内部接缝、水体、缺失和人为裁切边当接触带。
3. 所有基础因子从原始数据重算，在相同积分支撑和验证划分下比较旧/新接触距离，避免混淆算法改变和验证改变。
4. 重复空间分块验证之外，增加四个连续区域及50 km隔离验证，并输出每折地图。
5. 增加逐矿点影响、空间块留出靶区稳定性、区域事件数残差和测试特征超范围审计。

几何不会按任意容差吸附；原有图斑缝隙的影响需要结合几何审计判断。
5 km陆地面积支撑仍为旧中心筛选近似，本轮未声称实现精确海岸面积。
16点已在历史中用于方案比较，只作探索性评价。模型是目录事件相对强度，不是未知矿床数量或绝对有矿概率。

## 矿点审计

详见 `output/paper_audit/60矿点审计说明.md`。完整保留60点，标记未匹配、匹配偏远和候选距离接近的8点。新增验证与诊断不会根据外部预测得分更换矿点。
