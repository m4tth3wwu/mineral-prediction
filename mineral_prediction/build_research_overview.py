from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'output/ppp_run_20260905_195450_443'
OUT=ROOT/'reports'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':14,'axes.unicode_minus':False,'svg.fonttype':'path'})
ink='#26343c'; teal='#277878'; red='#a43c37'; gray='#66727a'; blue='#416b93'; pink='#ffbebe'
fig=plt.figure(figsize=(20,17),facecolor='white')
fig.text(.045,.965,'斑岩铜矿预测：PPP 方法改进与空间验证',fontsize=29,fontweight='bold',color=ink)
fig.text(.045,.935,'保留输入矿点位置 · 规范岩性接触距离 · 检查跨区域预测能力',fontsize=17,color=gray)
fig.text(.955,.942,'研究进展总览\n结果：2026-09-05 运行',fontsize=12,color=gray,ha='right')

def panel(pos,title):
 ax=fig.add_axes(pos);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 ax.add_patch(Rectangle((0,.87),1,.13,facecolor='#f1f2f2',edgecolor='none'))
 ax.text(.018,.936,title,fontsize=18,fontweight='bold',color=ink,va='center')
 return ax

def text(ax,x,y,s,size=14,color=ink,**kw):ax.text(x,y,s,fontsize=size,color=color,va='top',linespacing=1.6,**kw)
def box(ax,x,y,w,h,s,fc='#f5f8f8',size=14):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='#c7d0d2',lw=1))
 ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,color=ink,linespacing=1.5)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=gray,lw=1.4))
A=panel([.045,.685,.435,.225],'A  数据与研究框架')
text(A,.025,.81,'论文图件匹配矿点：60个；保持原44 / 16划分',16)
box(A,.025,.48,.42,.22,'44个训练点\n内部空间交叉验证');box(A,.55,.48,.42,.22,'16个历史验证点\n固定用于探索性比较')
text(A,.025,.43,'地化：Cu、Au、Mo、Fe  ｜  断层距离  ｜  重力\n岩性二分类  ｜  两类岩性接触边界距离',14)
box(A,.025,.03,.42,.19,'PPP：预测目录事件强度 λ',size=14);arrow(A,(.45,.125),(.54,.125));box(A,.55,.03,.42,.19,'按 λ 排序 → 候选靶区',size=14)
B=panel([.52,.685,.435,.225],'B  改进一：矿点事件项与面积积分项分开')
text(B,.02,.81,'原处理：矿点归入网格后，以网格代表位置计算。',14)
for x in np.linspace(.025,.40,5):B.plot([x,x],[.23,.65],color='#d6dddd',lw=1)
for y in np.linspace(.23,.65,4):B.plot([.025,.40],[y,y],color='#d6dddd',lw=1)
B.scatter([.072,.166,.259],[.30,.44,.58],s=30,c=blue)
B.scatter([.26],[.35],s=115,c=red,marker='*',zorder=5)
arrow(B,(.275,.35),(.46,.41));text(B,.48,.47,'矿点自身位置：提取特征\n积分网格：估计研究区面积积分',14)
text(B,.02,.19,'目标函数：Σ log λ(矿点) − Σ 面积 × λ(积分点) − 惩罚',14)
text(B,.02,.08,'实例：Kelsey距最近积分点3.223 km，模型未挪动矿点。',12,color=gray)
C=panel([.045,.415,.435,.24],'C  改进二：岩性类别与接触带定义')
text(C,.025,.81,'粉色侵入岩 = 1；其他有效岩性 = 0。',16)
for x in [.035,.205]:C.add_patch(Rectangle((x,.34),.17,.31,facecolor=pink,edgecolor=gray,lw=1))
C.add_patch(Rectangle((.375,.34),.20,.31,facecolor='#e7e6d8',edgecolor=gray,lw=1))
C.plot([.375,.375],[.34,.65],color=red,lw=4)
text(C,.625,.64,'同类图斑先合并\n只保留两类共同边界\n再计算矿点到边界的距离',14)
C.annotate('同类内部接缝不作为接触带',xy=(.205,.49),xytext=(.03,.25),fontsize=12,color=gray,arrowprops={'arrowstyle':'->','color':gray})
text(C,.025,.14,'水体、缺失及人为裁切边不构造地质接触带。\n新旧边界表现接近：改善定义，不宣称显著性能提升。',13)
D=panel([.52,.415,.435,.24],'D  改进三：带隔离的空间交叉验证')
text(D,.025,.81,'① 200 km空间块 × 五折 × 三次重复（种子42 / 43 / 44）\n② 四个连续区域留区验证（坐标聚类 + Voronoi分区）',14)
text(D,.025,.60,'两种验证均设50 km隔离；每折重新预处理、拟合。',14)
for x,w,col,label in [(.025,.55,teal,'28 训练'),(.575,.177,red,'9 测试'),(.752,.138,gray,'7 隔离')]:
 D.add_patch(Rectangle((x,.35),w,.14,facecolor=col));D.text(x+w/2,.42,label,color='white',fontsize=14,va='center',ha='center')
text(D,.025,.29,'实例：种子42第1折，44点分成28训练 + 9测试 + 7隔离。',13)
text(D,.025,.17,'该折最近训练—测试距离75.53 km；重拟合分数一致。\n16个历史验证点不进入这套内部交叉验证。',13)
E=panel([.045,.13,.435,.255],'E  特征对照结果：两种空间验证')
a=pd.read_csv(RUN/'spatial_cv_summary.csv').set_index('model');b=pd.read_csv(RUN/'compact_cv_summary.csv').set_index('model');c=pd.read_csv(RUN/'external_validation_summary.csv').set_index('model')
keys=['M0_no_lithology','M1_intrusive_indicator','M2_binary_contact','M2_old_polygon_contact','M3_coverage_sensitivity']
labels=['M0 基础因子','M1 + 岩性类别','M2 + 两类接触距离','对照：旧图斑边界','M3 + 覆盖代理']
chart=fig.add_axes([.177,.185,.275,.153])
for i,k in enumerate(keys):
 v=float(a.loc[k,'auc_mean']);lo=float(a.loc[k,'auc_min']);hi=float(a.loc[k,'auc_max'])
 chart.errorbar(v,4-i+.13,xerr=[[v-lo],[hi-v]],fmt='o',color=teal,capsize=3,markersize=6)
 chart.plot(float(b.loc[k,'auc_mean']),4-i-.13,'s',color=blue,markersize=6)
 chart.text(v+.004,4-i+.2,f'{v:.3f}',fontsize=11,color=teal)
 chart.text(float(b.loc[k,'auc_mean'])+.004,4-i-.28,f"{float(b.loc[k,'auc_mean']):.3f}",fontsize=11,color=blue)
chart.set_yticks(range(4,-1,-1),labels,fontsize=12);chart.set_xlim(.79,1.005);chart.set_ylim(-.6,4.7);chart.set_xticks([.8,.85,.9,.95,1]);chart.set_xlabel('存在点—背景 AUC（局部坐标轴）',fontsize=12);chart.grid(axis='x',alpha=.25);chart.spines[['top','right','left']].set_visible(False)
E.text(.025,.81,'● 重复分块汇总     ■ 连续区域汇总',fontsize=13,color=ink)
text(E,.025,.055,'横线为三次分块汇总值的最小—最大范围，不是置信区间。',11,color=gray)
F=panel([.52,.13,.435,.255],'F  历史验证与核验：结果如何解释')
text(F,.025,.81,'全区预测强度最高的约10%面积内，覆盖的验证点：',14)
for i,(name,key) in enumerate(zip(['M0 基础','M1 岩性','M2 接触距离','旧边界对照','M3 覆盖代理'],keys)):
 x=.025+i*.19
 F.text(x,.64,name,fontsize=11,color=gray)
 F.text(x,.51,f"{int(c.loc[key,'top_10pct_hits'])}/16",fontsize=23,color=teal if i==2 else ink,fontweight='bold')
text(F,.025,.39,'主模型连续区逐区 AUC：0.881–0.972，存在区域差异。',13)
text(F,.025,.28,'核验通过：60点输入坐标一致；19个主模型空间划分；\n26项来源 / 缓存哈希；1折重新拟合预测误差约1e-14。',13)
text(F,.025,.11,'4个未覆盖点：Fish Creek、Gabbs Group、\nSan Xavier North、Two Peaks；未因此删点或调阈值。',12,color=gray)
fig.add_artist(plt.Line2D([.045,.955],[.108,.108],color='#abb5b8',lw=1))
fig.text(.045,.085,'结论边界',fontsize=15,fontweight='bold',color=red)
fig.text(.13,.086,'接触距离在当前实验中提供预测信息；特征对照不能单独证明坐标改动或新验证带来的性能提升。',fontsize=13,color=ink)
fig.text(.13,.063,'λ不是直接的有矿概率；16点不是新盲测；输入位置保留不等于身份核实；勘查偏差及网格面积近似仍存在。',fontsize=12,color=gray)
fig.text(.045,.034,'方法参考：Renner等（2015）点过程与积分；Roberts等（2017）结构化验证；Phillips等（2009）抽样偏差。',fontsize=10,color=gray)
fig.text(.045,.018,'数据来源：ppp_run_20260905_195450_443 的空间验证汇总、外部验证汇总及 verification 核验记录。示意图不代表实际矿区几何。',fontsize=10,color=gray)
for ext in ['png','svg']:fig.savefig(OUT/f'PPP_科研总览_改进与结果.{ext}',dpi=240,facecolor='white')
print('Saved overview PNG / SVG')
