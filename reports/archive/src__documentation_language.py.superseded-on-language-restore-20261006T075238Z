"""Parse localized Markdown and verify explicitly archived editorial snapshots.

CSV, configuration, predictions and historical artifacts never use this exception.
English identifiers are internal parsing aliases, not the public document language.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
MASTER=Path(__file__).resolve().parents[1]
STYLE=json.loads((MASTER/'DOCUMENT_STYLE.json').read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def language_record():
 files=sorted((MASTER/'provenance').glob('MARKDOWN_LANGUAGE_STANDARDIZATION_*.json'))
 return json.loads(files[-1].read_text()) if files else None

def verify_snapshot(path,expected):
 path=Path(path).resolve()
 if sha(path)==expected:return True
 record=language_record()
 assert record and record['user_authorized'] and record['status'] in ['READY_FOR_VALIDATION','PASS'],path
 assert path.suffix=='.md' and path.is_relative_to(ROOT),path
 rel=str(path.relative_to(ROOT));entry=next((x for x in record['documents'] if x['path']==rel),None)
 assert entry and sha(path)==entry['after_sha256'],rel
 assert sha(ROOT/entry['archive'])==entry['before_sha256'],rel
 if expected==entry['before_sha256']:return True
 prior=next((x for x in record.get('prior_snapshots',[]) if x['path']==rel and x['sha256']==expected),None)
 assert prior and sha(ROOT/prior['archive'])==expected,(rel,'unknown historical snapshot')
 return True

def localize_prose(s):
 # Match the editorial prose transformation, leaving code/targets untouched.
 result=[];fence=False
 for line in s.splitlines():
  if line.startswith('```'):fence=not fence;result.append(line);continue
  if fence:result.append(line);continue
  pieces=re.split(r'(`[^`]*`|\]\([^)]*\)|\{\{.*?\}\})',line)
  for i,piece in enumerate(pieces):
   if i%2:continue
   piece=re.sub(r'\bCase ([1-5])',r'กรณี \1',piece)
   # Original pass and follow-up pass are separately recorded to avoid
   # cascading generic replacements through already translated technical terms.
   for translations in STYLE.get('translation_stages',[STYLE['prose_translations']]):
    for a,b in sorted(translations.items(),key=lambda x:-len(x[0])):
     piece=re.sub(r'(?<![A-Za-z_])'+re.escape(a)+r'(?![A-Za-z_])',lambda m:b,piece)
   pieces[i]=piece
  result.append(''.join(pieces))
 return '\n'.join(result)

def canonical_markup(s,path):
 name=Path(path).name
 if 'RESULTS_SUMMARY_TH' in name:allowed={'Winner ของแต่ละด้าน','Trade-off หลัก','Accuracy vs Speed','Accuracy vs Memory'}
 elif 'PRESENTATION_SUMMARY_TH' in name:allowed={'Failure Analysis','Near-tie visual check'}
 elif 'REPORT' in name:allowed={k for k in STYLE['headings'] if k[:1].isdigit() or k=='Qualitative Analysis'}
 else:allowed={'Overview','Models','Experimental Status','Main Result','Reports','Study Navigation','Reproducibility'}
 inverse={v:k for k,v in STYLE['headings'].items() if k in allowed}
 labels={'โมเดล':'Model','ตระกูล':'Family','ขนาด':'Tier','จำนวนพารามิเตอร์':'Parameters',
         'Checkpoint (MB)':'Checkpoint MB','Inference (ms)':'Inference ms','Pipeline (ms)':'Pipeline ms',
         'Peak allocated VRAM (MiB)':'Peak VRAM MiB','ผลลัพธ์':'Value','สิ่งที่ให้ความสำคัญ':'Priority',
         'โมเดลที่พิจารณา':'Candidate','หลักฐาน':'Evidence','รายการ':'Item','สถานะ':'Status',
         'ข้อมูล':'Dataset','ตัวประเมิน':'Evaluator','การเตรียมภาพ':'Preprocessing',
         'ขนาดภาพเข้าโมเดล':'Input size','ความละเอียดเชิงตัวเลข':'Precision','ค่าเกณฑ์':'Thresholds',
         'วิธีวัดเวลา':'Timing protocol','สภาพแวดล้อม':'Environment',
         'Mask mAP50-95 สูงสุด':'Highest Mask mAP50-95','AP75 สูงสุด':'Highest AP75','Recall สูงสุด':'Highest Recall',
         'Inference เร็วสุด':'Fastest inference','Pipeline เร็วสุด':'Fastest pipeline','FPS สูงสุด':'Highest FPS','VRAM ต่ำสุด':'Lowest VRAM'}
 if Path(path).name=='RESULTS_SUMMARY_TH.md':labels.update({'Inference เร็วสุด':'Inference speed','Pipeline เร็วสุด':'Pipeline speed'})
 out=[];fence=False
 for line in s.splitlines():
  if line.startswith('```'):fence=not fence;out.append(line);continue
  if fence:out.append(line);continue
  match=re.match(r'^(#{1,3}) (.*)$',line)
  if match:
   title=inverse.get(match[2],match[2]);title=re.sub(r'^กรณี (\S+) —',r'Case \1 —',title);title=re.sub(r'^ข้อสังเกต (\S+)',r'Observation \1',title)
   title=title.replace('การวิเคราะห์ภาพและพฤติกรรมเชิงคุณภาพ','Visual and Qualitative Analysis');line=match[1]+' '+title
  if line.startswith('|'):
   cells=line.split('|');line='|'.join(' '+labels.get(c.strip(),c.strip())+' ' if c.strip() else c for c in cells)
  line=line.replace('**การตีความ:**','**Interpretation:**').replace('FN; IoU สูงสุด ','FN; best IoU ').replace('[CSV มาตรฐาน]','[canonical CSV]')
  out.append(line)
 return '\n'.join(out)+'\n'
