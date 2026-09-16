"""Read-only source/path checks and a human-readable paper-point audit."""
import ast
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT=Path(__file__).resolve().parent
DATA=Path('D:/code/ResearchPractice')
OUT=ROOT/'output/paper_audit'
OUT.mkdir(exist_ok=True)
matched=pd.read_csv(ROOT/'output/paper_point_matching/paper_figure2g_star_matches.csv')
matched['candidate_gap_km']=matched.second_distance_km-matched.match_distance_km
matched['review_reason']=''
matched.loc[matched.match_status.ne('matched'),'review_reason']='未匹配；当前保留图上近似位置'
matched.loc[matched.match_status.eq('matched')&matched.match_distance_km.gt(10),'review_reason']='匹配距离超过10 km'
matched.loc[matched.match_status.eq('matched')&matched.candidate_gap_km.lt(1),'review_reason']='两候选距离差不足1 km'
matched.to_csv(OUT/'all_60_matching_audit.csv',index=False,encoding='utf-8-sig')
flags=matched.loc[matched.review_reason.ne('')]
allpoints=pd.read_csv(ROOT/'output/paper_point_matching/paper_figure2g_all_60_points_model_ready.csv')
train=allpoints.loc[allpoints.paper_split.eq('training')]
external=allpoints.loc[allpoints.paper_split.eq('validation')]
tree=BallTree(np.radians(train[['LATITUDE','LONGITUDE']].to_numpy(float)),metric='haversine')
d,i=tree.query(np.radians(external[['LATITUDE','LONGITUDE']].to_numpy(float)),k=1)
proximity=external[['DEPOSIT','paper_star_id']].copy()
proximity['nearest_training_deposit']=train.iloc[i.ravel()].DEPOSIT.to_numpy()
proximity['distance_km']=d.ravel()*6371.0088
proximity.to_csv(OUT/'external_training_proximity.csv',index=False,encoding='utf-8-sig')
lines=['# 60个论文重建矿点：来源与匹配审计','',
       '结论：可以保留作论文图件重建和方法实验；不能称为作者提供的原始60点坐标表，也不足以代表完整区域矿床清查。',
       '本轮不替换、不删除矿点，不根据模型预测好坏选择样本。','',
       '## 已核实来源','',
       '- 读取本地原论文 g53947.pdf 与补充材料 G53947_SuppMat.docx 的文本；检查现有Figure 2G匹配叠加图。',
       '- 现有脚本从613×588像素图件识别60个星标，按颜色重建44/16划分，再按最近邻匹配USGS候选目录。',
       '- 59个点取得目录匹配；1个未匹配，使用图件配准近似位置。训练集中43个为目录坐标，1个为近似坐标。',
       '- 57个high与2个medium是脚本按距离生成的标签，不是人工核实或统计置信度。',
       '- 本地已读正文和补充材料文本没有提供可逐项核对的60点名称坐标表；本轮未取得作者原始点表。','',
       '## 需要复核的8点','',
       '|图上编号|划分|当前最近候选（未匹配时不代表采用）|距图上位置/km|第二候选|两候选距离差/km|原因|',
       '|---|---|---|---:|---|---:|---|']
for _,r in flags.iterrows():
    lines.append(f'|{r.paper_star_id}|{r.paper_split}|{r.nearest_usgs_name}|{r.match_distance_km:.2f}|{r.second_usgs_name}|{r.candidate_gap_km:.2f}|{r.review_reason}|')
lines+=['','上述阈值用于排查，不是误差分布推导的置信阈值；近邻歧义不等于已经证明配错。',
        '编号22的最近候选Buckskin距离约185 km，未采用该候选，当前记录名称为Figure2G_unmatched_star_22。',
        '两个候选在图上的距离接近，可能对应相邻矿体/矿区，也可能配错；应核对矿区名称和原始坐标，而不是自动改成第二候选。','',
        '## 训练与外部点的空间独立性','',
        f'16个外部点中，距最近训练点小于50 km的有 {(proximity.distance_km<50).sum()} 个，小于10 km的有 {(proximity.distance_km<10).sum()} 个。',
        '这进一步说明16点不等于跨区独立验证；它们没有进入本轮拟合，但已在历史上参与多轮方案比较。',
        '全表见 external_training_proximity.csv。','',
        '## 建议','',
        '保留这60点为可追溯的论文重建版本；在取得更可靠来源后另建修订版本并比较，保留修改证据。',
        '优先找作者原始点表或更明确的矿区资料；不要仅为提高AUC剔除难预测矿点。',
        '若目标升级为区域找矿，需另行定义公开矿床目录的纳入条件、去重规则和真正未用于选方案的验证集。']
(OUT/'60矿点审计说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Inspect executable source, preserving historic result configs as provenance.
paths=[]
for file in list(ROOT.glob('*.py'))+list((DATA/'scripts').glob('*.py')):
    if file.name in {'migrate_research_paths.py','audit_project_delivery.py'}:continue
    source=file.read_text(encoding='utf-8-sig')
    tree=ast.parse(source,filename=str(file))
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if value.startswith(('D:\\code\\ResearchPractice\\','D:/code/ResearchPractice/')) and '\n' not in value:
                paths.append({'script':str(file),'path':value,'exists':Path(value).exists()})
pd.DataFrame(paths).to_csv(OUT/'executable_path_audit.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'flagged_points':len(flags),'external_within_50km':int((proximity.distance_km<50).sum()),
                  'missing_literal_paths':[r for r in paths if not r['exists']]},ensure_ascii=True,indent=2))
