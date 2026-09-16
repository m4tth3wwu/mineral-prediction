# 当前矿产预测项目入口

主工程已迁入 D:\code\ResearchPractice\mineral_prediction，代码、已有结果、文献、数据子目录和Git元数据均保留。

在PowerShell运行：

```powershell
& 'D:\code\ResearchPractice\run_current_ppp.ps1'
```

启动脚本使用原44训练点/16验证点，并自动创建新的 output/ppp_run_时间戳 目录。
首次迁移后运行可能因路径及代码哈希变化自动重建特征缓存；历史结果文件保持原样。
选点对照实验入口为 mineral_prediction/select_mineral_points.py，不是默认主程序。
C盘原目录暂保留为备份；以后请打开D盘主工程。复制内容核验清单见 mineral_prediction/migration_verification.json。
