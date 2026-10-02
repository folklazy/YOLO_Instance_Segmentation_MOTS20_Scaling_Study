"""Plot only canonical preserved results; never run inference."""
from stage0_standardize import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=['#2563eb','#ea580c','#16a34a','#9333ea']
for i in range(2):
 p=ROOT/NAMES[i];data=rows(p/'metrics/TIER_RESULTS.csv');names=[r['model'] for r in data];out=p/'outputs/plots';out.mkdir(parents=True,exist_ok=True)
 def bar(filename,key,ylabel):
  fig,ax=plt.subplots(figsize=(8,4.8));v=[float(r[key]) for r in data];ax.bar(names,v,color=COLORS)
  ax.set_ylabel(ylabel);ax.set_title(LABELS[i]+' — MOTS20');ax.set_ylim(0,max(v)*1.18)
  for j,r in enumerate(data):ax.text(j,v[j],fmt(key,r[key]),ha='center',va='bottom',fontsize=9)
  fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
 bar('01_mask_map50_95.png','mask_map50_95','Mask mAP50-95 (0–1)')
 fig,ax=plt.subplots(figsize=(8,4.8));x=list(range(4))
 for offset,key,col in [(-.2,'ap50','#2563eb'),(.2,'ap75','#ea580c')]:ax.bar([j+offset for j in x],[float(r[key]) for r in data],width=.4,label=key.upper(),color=col)
 ax.set_xticks(x,names);ax.set_ylim(0,1);ax.set_ylabel('AP (0–1)');ax.set_title(LABELS[i]+' — MOTS20');ax.legend();fig.tight_layout();fig.savefig(out/'02_ap50_ap75.png',dpi=160);plt.close(fig)
 bar('03_inference_latency.png','inference_ms_mean','Inference mean (ms/frame)')
 bar('04_pipeline_fps.png','fps','Pipeline FPS (excludes RLE / disk I/O)')
 bar('05_peak_vram.png','peak_allocated_vram_mib','Peak allocated VRAM (MiB)')
 fig,ax=plt.subplots(figsize=(8,5.2))
 for j,r in enumerate(data):
  ax.scatter(float(r['inference_ms_mean']),float(r['mask_map50_95']),color=COLORS[j],label=r['model'],s=80)
 ax.set_xlabel('Inference mean (ms/frame)');ax.set_ylabel('Mask mAP50-95 (0–1)');ax.set_title(LABELS[i]+' — MOTS20');ax.legend(loc='best');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'06_accuracy_vs_latency.png',dpi=160);plt.close(fig)
 files=sorted(out.glob('0*.png'))
 write(out/'INDEX.md','# Canonical tier plots\n\nGenerated only from [TIER_RESULTS.csv](../../metrics/TIER_RESULTS.csv), preserving fixed model order and saved measurements. Historical run-scoped plots remain intact.\n\n'+'\n'.join(f'- [{x.name}]({x.name})' for x in files)+'\n')
 dump(p/'manifests/PLOT_PROVENANCE.json',{'source':'metrics/TIER_RESULTS.csv','source_sha256':sha(p/'metrics/TIER_RESULTS.csv'),'generator':'../'+MASTER.name+'/scripts/plot_tiers.py','generator_sha256':sha(Path(__file__)),'timestamp':NOW,'inference_rerun':False,'outputs':{str(x.relative_to(p)):sha(x) for x in files}})
print('12 canonical plots generated from preserved CSVs.')
