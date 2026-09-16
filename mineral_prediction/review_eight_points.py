"""Read-only source audit and deterministic pixel sensitivity; never edits labels."""
from pathlib import Path
import csv, hashlib, json, math
from collections import Counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/point_review_20260905'
IDS = [22, 26, 27, 39, 40, 44, 48, 56]
def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def distance(lon, lat, x, y):
    a,b=map(math.radians,[lat,y]); dl=math.radians(x-lon); da=b-a
    return 6371.0088*2*math.asin(min(1,math.sqrt(math.sin(da/2)**2+math.cos(a)*math.cos(b)*math.sin(dl/2)**2)))
def write(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=list((ROOT/'output/paper_point_matching').glob('*.csv'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    stars=read(ROOT/'output/paper_point_matching/paper_figure2g_star_matches.csv')
    cat=read(ROOT/'data/usgs_porpyhry/western_us_deposit_points_used.csv')
    urls={r['file']:r['url'] for r in json.loads((OUT/'sources/download_manifest.json').read_text())}
    facts={
      22:('新增强候选，作者身份未确认','Moonlight 技术报告给出矿床近似中心 40°13′36″N、120°48′11″W；原目录最近的 Buckskin 距离约185公里不能作为替代。','moonlight_2018.pdf','PDF第42页，正文4-1','原始图上坐标保留；Moonlight 单独列为待确认候选。'),
      26:('矿区内候选未定','MacArthur 技术报告将 Bear-MacArthur-Lagomarsino 描述为大型斑岩铜系统，其一部分位于 Yerington 场地下；不能据此把 Weed Heights 与 Bear 当作同义词。','https://www.sec.gov/Archives/edgar/data/1339688/000106299314000353/exhibit99-1.htm','24.2节，正文209页','保留原匹配，标记近邻歧义。'),
      27:('项目与矿体层级重叠','Gabbs 报告明确将 Sullivan 列为项目内四个矿床之一；Gabbs Group 与 Sullivan 可能是不同层级记录，不能仅因距离接近认定配错或视为独立矿床。','gabbs_2024.pdf','PDF第20页，正文1-1','保留原点；增加 Gabbs/Sullivan 层级关系标记，暂不合并目录。'),
      39:('位置偏差及类型待核实','Nickerson与Seedorff（2016）确认 Sheep Mountain 存在斑岩系统，摘要提出可能属于 Mo-Cu 子类；这支持地质合理性，但不确认本图星号身份或目录坐标。','https://experts.arizona.edu/en/publications/dismembered-porphyry-systems-near-wickenburg-arizona-district-sca/','作者机构收录的论文摘要；全文下载404','保留原匹配；列为优先核对原始坐标和研究矿种范围的点。'),
      40:('较大位置偏差未解决','运营方资料确认 Miami 位于 Globe-Miami 矿区且为斑岩铜矿；该事实不足以解释论文星号与目录坐标约15公里的偏差。','https://fcx.com/operations/north-america','Miami 的 Location 与 Ores','保留原匹配；不能因邻矿存在就替换。'),
      44:('两个真实近邻矿点，图上身份未定','USGS 报告将 Christmas 与 Chilito 分列为 Banner 矿区内不同矿山，并描述 Chilito 在 Christmas 以西约4公里。','usgs_christmas_1979.pdf','PDF第14、67页','保留原匹配；Christmas/Chilito 都列入待核对候选。'),
      48:('矿区内部身份与类型待核实','新墨西哥地质局摘要把 Continental 放在 Fierro-Hanover 矿区北部，讨论接触变质与矽卡岩矿化；不能直接证明 Continental Breccia Pipe 与 Hanover Mountain 是同一个斑岩矿点。','https://geoinfo.nmt.edu/museum/nmms/abstracts/view.cfm?aid=173','Eveleth（1995）摘要，第4页','保留原匹配；核对矿体层级及类型，勿把矿区名称等同具体矿体。'),
      56:('近邻且可能同源，图上身份未定','Cook与Porter（2005）分别描述 San Xavier North 和 Mission-Pima，并引述两者等可能来自同一原始系统的解释；同源假说不等于同一个当前坐标。','san_xavier_2005.pdf','PDF第15页，正文221页','保留原匹配；记录同源假说，等待作者原始点表。')
    }
    rows=[]; sensitivity=[]; candidate=[]
    for s in stars:
        i=int(s['paper_star_id'])
        if i not in IDS: continue
        lon=float(s['paper_longitude_approx']); lat=float(s['paper_latitude_approx'])
        status,fact,source,page,action=facts[i]
        source=urls.get(source,source)
        rows.append(dict(paper_star_id=i,split=s['paper_split'],original_match=s['nearest_usgs_name'] if s['match_status']=='matched' else '未匹配（保留图上坐标）',original_distance_km=s['match_distance_km'],review_status=status,verified_context=fact,source_url=source,source_location=page,action=action,author_identity_confirmed=False))
        # Integer offsets are controlled scenarios, not estimated errors or probabilities.
        for radius in (1,2):
            counts=Counter()
            for dx in range(-radius,radius+1):
                for dy in range(-radius,radius+1):
                    x=lon+dx/23.16; y=lat-dy/23.2
                    nearest=min(cat,key=lambda c:distance(x,y,float(c['deposit_longitude']),float(c['deposit_latitude'])))
                    counts[nearest['DEPOSIT']]+=1
            sensitivity.append(dict(paper_star_id=i,offset_radius_pixels=radius,scenario_count=(2*radius+1)**2,distinct_nearest_candidates=len(counts),scenario_counts=json.dumps(counts,ensure_ascii=False),interpretation='确定性扰动场景计数，非概率；独立最近邻，不是原始全局唯一匹配'))
        if i==22:
            x=-(120+48/60+11/3600); y=40+13/60+36/3600
            candidate.append(dict(paper_star_id=i,candidate='Moonlight',longitude=x,latitude=y,distance_to_figure_star_km=distance(lon,lat,x,y),source_url=source,source_location=page,status='候选，未经论文作者确认，未用于训练',coordinate_definition='报告的矿床近似中心；非测量级定位'))
    write('eight_point_review.csv',rows);write('pixel_sensitivity.csv',sensitivity);write('candidate_additions_NOT_model_input.csv',candidate)
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'original_input_hashes.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# 8个矿点逐项复核（2026-09-05）','',
    '结论：没有依据判定这8个点全部选错，也没有依据把原匹配标记 high 当作身份已经核实。本轮发现一个有一手坐标支持的新候选 Moonlight，识别了一处明确的项目—矿体层级关系，其余仍需原始点表或更高分辨率图件。原60点、44/16划分和模型输入均未更改。','',
    f"第22号星号距 Moonlight 报告近似中心约 **{candidate[0]['distance_to_figure_star_km']:.2f} km**。这是强候选，尚不能宣布找回论文原始矿点。",'',
    '## 逐项结果','', '|星号|原匹配|复核判断|', '|---|---|---|']
    md += [f"|{r['paper_star_id']}（{r['split']}）|{r['original_match']}|{r['review_status']}|" for r in rows]
    for r in rows:
        md += ['',f"### {r['paper_star_id']}：{r['original_match']}",'',r['verified_context'],f"依据：[原始来源]({r['source_url']})，{r['source_location']}。",'',f"处理：{r['action']}"]
    md += ['', '## 图件分辨率与匹配标签','',
    '沿用现有标定：x=23.16×经度+2943.2，y=−23.2×纬度+1229.0。一像素南北约4.79公里，东西方向随纬度变化。距离差不足1公里的两个候选，通常不能靠该图的最近邻距离判定身份。原 high/medium 是10公里阈值分档，不是统计置信度，也不是文献身份核实。','',
    'pixel_sensitivity.csv 对原目录做±1、±2像素整数网格扰动，分别9和25个场景。计数只说明在所选扰动下最近候选是否变化，不表示正确概率；没有模拟标定整体误差或原脚本的全局唯一分配。Moonlight 未加入该实验目录，故第22号场景仍只反映不完整旧目录。','',
    '## 可追溯性与下一步','',
    '已下载并提取5份原始PDF，来源及SHA256见 sources/download_manifest.json。Sheep Mountain 全文链接返回404，本轮仅据作者机构的摘要作有限判断。网页来源在逐项记录中给出。',
    '已核对 Figshare 条目30888581：列出的补充文件为 G53947_SuppMat.docx；本地文件MD5与条目 bb47b441ebfb9755b1ce5bf1b10a2679 一致。此前提取的正文与补充材料未发现60点逐点名称坐标表；这不能证明作者没有其他未公开数据。',
    '建议优先取得作者原始名称—坐标—训练/验证标签表；在取得之前，把本结果用于透明报告及后续敏感性试验。不要根据模型分数高低选择候选，也不要自动删除点或改动外部验证标签。',
    '原始CSV哈希见 original_input_hashes.json，脚本执行前后核对一致。候选文件刻意命名为 NOT_model_input，当前模型不读取它。','']
    (OUT/'8矿点逐项复核.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'reviewed':len(rows),'moonlight_distance_km':candidate[0]['distance_to_figure_star_km'],'original_csv_unchanged':True},ensure_ascii=True))

if __name__=='__main__': main()
