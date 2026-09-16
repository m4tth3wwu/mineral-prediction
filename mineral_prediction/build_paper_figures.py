from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.colors import ListedColormap
R=Path(__file__).resolve().parent;S=R/'output/ppp_run_20260905_195450_443';O=R/'reports/paper_figures';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
def save(fig,n):
 for e in ['png','svg']:fig.savefig(O/(n+'.'+e),dpi=350,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,t,mark=False):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='#eaf3f3' if mark else 'white',edgecolor='#346970' if mark else '#555',lw=1))
 ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11,linespacing=1.6)
def arr(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#555'))
for x,t in [(.02,'矿点记录（60）\n训练44 / 历史验证16'),(.375,'区域协变量\n地化、断层、重力、岩性'),(.73,'固定面积积分支撑\n5 km支撑；10 km拟合积分')]:box(x,.8,.25,.15,t)
for x,t,m in [(.02,'保留输入矿点位置\n矿点处提取特征',True),(.375,'岩性二分类与共同边界\n计算接触距离',True),(.73,'积分点处提取特征\n面积加权近似积分',False)]:box(x,.51,.25,.17,t,m);arr((x+.125,.8),(x+.125,.68));arr((x+.125,.51),(.5,.4))
box(.375,.25,.25,.15,'PPP模型拟合\n事件项 − 积分项 − 惩罚')
for x,t in [(.02,'内部空间交叉验证\n分块 / 连续区域 + 50 km隔离'),(.365,'固定16点历史验证\nAUC与前10%面积覆盖'),(.71,'全区预测强度排序\n预测有利区与稳定性')]:box(x,.02,.27,.15,t,x==.02);arr((.5,.25),(x+.135,.17))
save(fig,'Fig1_方法流程')
keys=['M0_no_lithology','M1_intrusive_indicator','M2_binary_contact','M2_old_polygon_contact','M3_coverage_sensitivity'];labels=['M0','M1','M2-new','M2-old','M3'];colors=['#7c8790','#82a5b4','#26767b','#bd9b74','#9684a5']
a=pd.read_csv(S/'spatial_cv_summary.csv').set_index('model');b=pd.read_csv(S/'compact_cv_summary.csv').set_index('model');c=pd.read_csv(S/'external_validation_summary.csv').set_index('model');r=pd.read_csv(S/'spatial_repeat_metrics.csv');f=pd.read_csv(S/'compact_fold_metrics.csv')
fig,axs=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
for j,(tab,title) in enumerate([(a,'(a) 重复空间分块验证'),(b,'(b) 连续区域留区验证')]):
 ax=axs[j];v=[tab.loc[k,'auc_mean'] for k in keys];ax.bar(range(5),v,color=colors,width=.6,alpha=.8)
 for i,k in enumerate(keys):
  pts=r[r.model.eq(k)].same_fold_auc.to_numpy() if j==0 else f[f.model.eq(k)].presence_background_auc.to_numpy()
  ax.scatter(i+np.linspace(-.15,.15,len(pts)),pts,s=17,facecolors='white',edgecolors='black',linewidths=.6,zorder=3);ax.text(i,1.025,f'{v[i]:.3f}',ha='center')
 ax.set(xticks=range(5),xticklabels=labels,ylim=(0,1.1),yticks=np.linspace(0,1,6),ylabel='存在点—背景 AUC',title=title)
ax=axs[2];hits=[int(c.loc[k,'top_10pct_hits']) for k in keys];ax.bar(range(5),np.array(hits)/16,color=colors,width=.6)
for i,n in enumerate(hits):ax.text(i,n/16+.025,f'{n}/16',ha='center')
ax.set(xticks=range(5),xticklabels=labels,ylim=(0,1.1),yticks=np.linspace(0,1,5),ylabel='历史验证点覆盖比例',title='(c) 前10%面积覆盖')
for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=9)
save(fig,'Fig2_模型性能对比')
g=pd.read_csv(S/'predictions_5km.csv');ev=pd.read_csv(R/'output/ppp_binary_features_v3/events.csv');ex=pd.read_csv(R/'output/ppp_binary_features_v3/external.csv');m=pd.read_csv(S/'verification/four_missed_points.csv')
v=g.M2_binary_contact_intensity.to_numpy();w=g.area_km2.to_numpy();order=np.argsort(v);sv=v[order];cw=np.cumsum(w[order]);rank=cw[np.searchsorted(sv,v,side='right')-1]/cw[-1];threshold=sv[np.searchsorted(cw,.9*cw[-1])]
xs=np.sort(g.metric_x.unique());ys=np.sort(g.metric_y.unique());assert len(xs)*len(ys)<2000000
ix=np.searchsorted(xs,g.metric_x);iy=np.searchsorted(ys,g.metric_y);z=np.full((len(ys),len(xs)),np.nan);t=z.copy();z[iy,ix]=rank*100;t[iy,ix]=(v>=threshold).astype(float)
fig,axs=plt.subplots(1,2,figsize=(12,7),layout='constrained')
im=axs[0].pcolormesh(xs/1000,ys/1000,z,shading='nearest',cmap='viridis',vmin=0,vmax=100,rasterized=True);axs[1].pcolormesh(xs/1000,ys/1000,t,shading='nearest',cmap=ListedColormap(['#e9ecec','#b65444']),vmin=0,vmax=1,rasterized=True)
for ax in axs:
 ax.scatter(ev.metric_x/1000,ev.metric_y/1000,s=14,c='black',label='训练点（44）',zorder=4);ax.scatter(ex.metric_x/1000,ex.metric_y/1000,s=32,marker='^',facecolors='white',edgecolors='black',linewidths=.8,label='历史验证点（16）',zorder=5)
 ax.set_aspect('equal');ax.set_xlabel('东向坐标（km）');ax.set_ylabel('北向坐标（km）');ax.tick_params(labelsize=9)
axs[0].set_title('(a) M2-new 预测强度面积百分位');axs[1].set_title('(b) 全区前10%预测有利区')
mp=ex[ex.paper_star_id.isin(m.paper_star_id)];axs[1].scatter(mp.metric_x/1000,mp.metric_y/1000,s=85,facecolors='none',edgecolors='#245bad',linewidths=1.2,label='未覆盖验证点（4）',zorder=6)
for q in mp.itertuples():
 offset={20:(-50,20),27:(-60,-18),56:(-60,25),58:(18,-22)}[int(q.paper_star_id)]
 axs[1].annotate(str(int(q.paper_star_id)),(q.metric_x/1000,q.metric_y/1000),xytext=offset,textcoords='offset points',fontsize=10,color='#245bad',arrowprops={'arrowstyle':'-','color':'#245bad','lw':.7},zorder=7)
axs[1].legend(loc='upper right',fontsize=8,framealpha=.95)
cb=fig.colorbar(im,ax=axs[0],orientation='horizontal',shrink=.78,pad=.04);cb.set_label('面积百分位（%）；不是有矿概率',fontsize=9)
axs[1].text(.02,.02,'红色：全区强度最高的约10%面积\nESRI:102039；5 km预测支撑',transform=axs[1].transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
save(fig,'Fig3_空间预测与验证矿点')
(O/'图注与使用说明.md').write_text("""# 图注

图1｜PPP预测研究流程。浅蓝框突出此次改进：保留输入矿点坐标、规范岩性接触距离及空间验证。矿点事件项与面积积分项分别计算，保留原44/16划分。内部验证每折重新预处理与拟合；流程箭头为概念表达，16个历史验证点不进入训练。

图2｜特征方案的性能对照。(a)柱为三次空间分块验证汇总AUC的平均，空心点为三次结果；(b)柱为连续区域的汇总AUC（按测试事件数加权），空心点为四个区域的AUC，柱高不一定等于四点的简单平均；(c)全区前10%面积对16个历史验证点的覆盖。点的散布不是置信区间。M0：基础地化、断层与重力因子；M1：增加岩性类别；M2-new：进一步增加两类有效岩性共同边界距离；M2-old：使用旧图斑边界距离；M3：在M2-new上增加采样覆盖代理。该对照不能单独证明坐标改动或验证设计改变带来的性能提升。

图3｜M2-new预测与矿点分布。(a)预测目录事件强度的面积加权百分位；(b)强度最高的约10%面积为红色。黑点为训练点，白三角为历史验证点，蓝圈为未被前10%覆盖的4点。星号编号20：Fish Creek；27：Gabbs Group；56：San Xavier North；58：Two Peaks。空白表示当前预测支撑之外，不是确定无矿。采用ESRI:102039等面积投影和5 km预测网格，无行政区划或精细海岸线。百分位不是有矿概率，16点不是新盲测。

数据源：ppp_run_20260905_195450_443及verification核验文件；未重新拟合或改动矿点。PNG为350 dpi。SVG中图1与图2为矢量；图3密集栅格背景嵌入为图像，文字、标记与坐标为矢量。图中中文可用于导师汇报，投稿需按期刊要求调整语言与尺寸。
""",encoding='utf-8')
print('Created 3 PNG/SVG pairs and captions')
