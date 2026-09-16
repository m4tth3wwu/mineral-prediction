"""Download public primary-source documents for the eight-point audit."""
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

OUT=Path(__file__).resolve().parent/'output/point_review_20260905/sources'
OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'moonlight_2018.pdf':'https://uscoppercorp.com/wp-content/uploads/2021/03/Moonlight-PEA-FINAL-12Apr18.pdf',
 'gabbs_2024.pdf':'https://p2gold.com/_resources/reports/GABBS-PROJECT_HEAP-LEACH-MILL-PEA-43-101_03July2024-SEDAR.pdf',
 'usgs_christmas_1979.pdf':'https://pubs.usgs.gov/of/1979/0844/report.pdf',
 'usgs_nevada_2009.pdf':'https://pubs.usgs.gov/of/2009/1271/of2009-1271.pdf',
 'san_xavier_2005.pdf':'https://portergeo.com.au/full_text/Cook_Porter_SW_NAmer_Supergene-PGC_Publishing.pdf',
 'sheep_mountain_2016.pdf':'https://www.geo.arizona.edu/sites/www.geo.arizona.edu/files/Nickerson%26Seedorff%2C2016%2CWickenburg%26ArcReconstruction_EGv111n2p447-466.pdf',
 'paper_supplement_metadata.json':'https://api.figshare.com/v2/articles/30888581',
}

def fetch(item):
 name,url=item
 try:
  path=OUT/name
  if path.exists(): data=path.read_bytes()
  else:
   with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as r:data=r.read()
   if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise ValueError('Not a PDF')
   path.write_bytes(data)
  return {'file':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded'}
 except Exception as e:return {'file':name,'url':url,'status':'failed','error':str(e)}

if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,SOURCES.items()))
 (OUT/'download_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(rows,ensure_ascii=True,indent=2))
