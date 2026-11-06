"""Durable full HoloAssist RGB/gaze/calibration download, below 400 GB.

Run outside OneDrive. Range chunks resume, ETags pin publisher versions, and
completed archives are assembled with SHA-256 records. No automatic extraction.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import requests

BASE='https://hl2data.z5.web.core.windows.net/holoassist-data-release/'
FILES={
    'data-annotation-trainval-v1_1.json':BASE+'data-annotation-trainval-v1_1.json',
    'data-splits-v1_2.zip':'https://holoassist.github.io/label_files/data-splits-v1_2.zip',
    'eyes.tar':BASE+'eyes.tar',
    'cam_info.tar':BASE+'cam_info.tar',
    'video_compress.tar':BASE+'video_compress.tar',
}
LOCAL=threading.local()


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def session():
    if not hasattr(LOCAL,'http'):
        LOCAL.http=requests.Session()
    return LOCAL.http


def discover(output):
    lock=output/'sources.json'
    if lock.exists():
        sources=json.loads(lock.read_text())
        if {r['name']:r['url'] for r in sources}!=FILES:
            raise RuntimeError('Existing source inventory differs')
        return sources
    sources=[]
    for name,url in FILES.items():
        while True:
            try:
                r=session().head(url,timeout=(10,30),allow_redirects=True)
                r.raise_for_status()
                size=int(r.headers['Content-Length'])
                if size<=0:
                    raise ValueError('Unknown source size')
                sources.append({'name':name,'url':url,'bytes':size,'etag':r.headers.get('ETag'),
                                'last_modified':r.headers.get('Last-Modified')})
                print('SOURCE',name,size,flush=True)
                break
            except Exception as exc:
                atomic_json(output/'status.json',{'state':'retrying_source_discovery','file':name,
                    'error':str(exc),'updated_unix':time.time()})
                print('Source retry',name,str(exc),flush=True)
                time.sleep(30)
    total=sum(r['bytes'] for r in sources)
    if total>400_000_000_000:
        raise RuntimeError('Refusing bundle above user 400 GB limit')
    if shutil.disk_usage(output).free<2*total+5_000_000_000:
        raise RuntimeError('Insufficient free space for archives plus resumable chunks')
    atomic_json(lock,sources)
    return sources


def fetch_chunk(source,index,start,end,folder):
    target=folder/f'{index:06d}.part'
    receipt=target.with_suffix('.json')
    expected=end-start+1
    if target.exists() and receipt.exists() and target.stat().st_size==expected:
        saved=json.loads(receipt.read_text())
        if saved.get('etag')==source['etag'] and saved.get('start')==start and saved.get('end')==end:
            return expected
    partial=target.with_suffix('.downloading')
    attempt=0
    while True:
        attempt+=1
        try:
            headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
            if source['etag']:
                headers['If-Match']=source['etag']
            with session().get(source['url'],headers=headers,stream=True,timeout=(10,90)) as r:
                if r.status_code==412:
                    raise RuntimeError('SOURCE_CHANGED: publisher ETag no longer matches')
                full_small=(start==0 and expected==source['bytes'] and r.status_code==200
                            and int(r.headers.get('Content-Length','-1'))==expected)
                ranged=(r.status_code==206 and r.headers.get('Content-Range')==f"bytes {start}-{end}/{source['bytes']}")
                if not (ranged or full_small):
                    raise ValueError(f'Unhonored range: {r.status_code}, {r.headers.get("Content-Range")}')
                digest=hashlib.sha256();count=0
                with partial.open('wb') as stream:
                    for block in r.iter_content(1024*1024):
                        if count+len(block)>expected:
                            raise ValueError('Oversized response')
                        stream.write(block);digest.update(block);count+=len(block)
                    stream.flush();os.fsync(stream.fileno())
                if count!=expected:
                    raise ValueError(f'Incomplete chunk {count}/{expected}')
            partial.replace(target)
            atomic_json(receipt,{'start':start,'end':end,'etag':source['etag'],'sha256':digest.hexdigest()})
            return expected
        except Exception as exc:
            if 'SOURCE_CHANGED:' in str(exc):
                raise
            print('Chunk retry',source['name'],index,attempt,str(exc),flush=True)
            time.sleep(min(60,5*attempt))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path.home()/'Datasets/HoloAssist')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--chunk-mib',type=int,default=32)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if 'onedrive' in str(output).lower():
        raise ValueError('Bulk destination must be outside OneDrive')
    status=output/'status.json';sources=discover(output)
    sources.sort(key=lambda source:source['name']!='video_compress.tar')
    total=sum(r['bytes'] for r in sources);done=0;started=time.time()
    print('TOTAL',total,'bytes; destination',output,flush=True)
    chunk=args.chunk_mib*1024*1024
    for source in sources:
        target=output/source['name'];verified=target.with_suffix(target.suffix+'.verified.json')
        if target.exists() and target.stat().st_size==source['bytes'] and verified.exists():
            saved=json.loads(verified.read_text())
            if saved.get('etag')==source['etag']:
                done+=source['bytes'];continue
        folder=output/'chunks'/source['name'];folder.mkdir(parents=True,exist_ok=True)
        tasks=[(i,start,min(source['bytes']-1,start+chunk-1))
               for i,start in enumerate(range(0,source['bytes'],chunk))]
        completed=0
        def run(task):
            return fetch_chunk(source,*task,folder)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                completed+=future.result()
                info={'state':'downloading','file':source['name'],'file_completed_bytes':completed,
                      'file_total_bytes':source['bytes'],'completed_bytes':done+completed,'total_bytes':total,
                      'completed_gb':round((done+completed)/1e9,3),'total_gb':round(total/1e9,3),
                      'percent':round(100*(done+completed)/total,3),'updated_unix':time.time(),
                      'destination':str(output),'workers':args.workers}
                atomic_json(status,info)
                print(json.dumps(info),flush=True)
        atomic_json(status,{'state':'assembling','file':source['name'],'completed_bytes':done+completed,
                            'total_bytes':total,'updated_unix':time.time()})
        assembled=target.with_suffix(target.suffix+'.assembling');digest=hashlib.sha256()
        with assembled.open('wb') as stream:
            for index,start,end in tasks:
                part=folder/f'{index:06d}.part'
                receipt=json.loads(part.with_suffix('.json').read_text());part_hash=hashlib.sha256()
                with part.open('rb') as data:
                    for block in iter(lambda:data.read(4*1024*1024),b''):
                        part_hash.update(block);digest.update(block);stream.write(block)
                if part_hash.hexdigest()!=receipt['sha256']:
                    raise RuntimeError(f'Chunk hash mismatch: {part}')
            stream.flush();os.fsync(stream.fileno())
        if assembled.stat().st_size!=source['bytes']:
            raise RuntimeError('Assembled archive size mismatch')
        assembled.replace(target)
        atomic_json(verified,{**source,'sha256':digest.hexdigest(),'verified_unix':time.time()})
        # Remove only generated chunk files after the assembled archive is verified.
        for index,_,_ in tasks:
            part=folder/f'{index:06d}.part';part.unlink();part.with_suffix('.json').unlink()
        done+=source['bytes']
        print('VERIFIED',source['name'],digest.hexdigest(),flush=True)
    atomic_json(status,{'state':'complete','completed_bytes':done,'total_bytes':total,
                        'destination':str(output),'elapsed_seconds':time.time()-started,'updated_unix':time.time()})
    print('COMPLETE',done,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('FATAL',repr(exc),flush=True)
        raise
