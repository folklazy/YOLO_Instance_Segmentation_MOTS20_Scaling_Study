"""Verify existing GitHub branches, tracked image blobs and real PNG HTTP responses."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from urllib.parse import quote
import json,re,subprocess,urllib.request

MASTER=Path(__file__).resolve().parents[1];ROOT=MASTER.parent


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,timeout=45).strip()


def verify_repo(name):
    root=ROOT/name;require=lambda c,m: c or (_ for _ in ()).throw(ValueError(m))
    require(git(root,'branch','--show-current')=='main','Unexpected branch')
    require(git(root,'remote','get-url','origin')==f'https://github.com/folklazy/{name}.git','Unexpected remote')
    head=git(root,'rev-parse','HEAD');remote=git(root,'ls-remote','origin','refs/heads/main').split()[0]
    require(head==remote,'Published HEAD mismatch: '+name)
    req=urllib.request.Request(f'https://api.github.com/repos/folklazy/{name}/git/trees/{head}?recursive=1',headers={'User-Agent':'MOTS20-post-study-publication'})
    with urllib.request.urlopen(req,timeout=30) as response:tree=json.load(response)
    require(not tree.get('truncated'),'Truncated remote tree')
    remote_tree={e['path']:e for e in tree['tree']};images=set()
    docs=git(root,'ls-files','*.md').splitlines()
    for rel in docs:
        if any(p in Path(rel).parts for p in ['archive','frozen_inputs','frozen_pilot','templates','report_templates']):continue
        text=re.sub(r'<!--.*?-->','',(root/rel).read_text(),flags=re.S)
        for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)',text):
            if link.startswith(('http','data:')):continue
            path=(root/rel).parent/link.split('#')[0];require(path.resolve().is_relative_to(root.resolve()),'Image outside owning repo')
            images.add(str(path.resolve().relative_to(root.resolve())))
    verified=[]
    for rel in sorted(images):
        require(git(root,'ls-files','--error-unmatch',rel)==rel,'Untracked image '+rel)
        blob=git(root,'hash-object',rel)
        require(remote_tree[rel]['type']=='blob' and remote_tree[rel]['sha']==blob,'Remote image blob mismatch '+rel)
        url=f'https://raw.githubusercontent.com/folklazy/{name}/main/{quote(rel)}'
        request=urllib.request.Request(url,headers={'Range':'bytes=0-31','User-Agent':'MOTS20-post-study-publication'})
        with urllib.request.urlopen(request,timeout=30) as response:
            require(response.status in [200,206] and response.headers.get_content_type()=='image/png','Image HTTP type '+rel)
            require(response.read(8)==b'\x89PNG\r\n\x1a\n','Image HTTP bytes '+rel)
        verified.append({'path':rel,'git_blob_sha':blob,'remote_blob_matches':True,'http_png':'PASS'})
    return {'repository':name,'branch':'main','revision':head,'remote_head_matches':True,'images':verified}


def verify():
    revision=json.loads((MASTER/'provenance/POST_STUDY_REVISION.json').read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(verify_repo,list(revision['repositories'])))
    return {'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'repositories':results,'images_checked':sum(len(r['images']) for r in results),
            'scientific_status_independent_of_push_authentication':True}


if __name__=='__main__':print(json.dumps(verify(),indent=2)+'\n')
