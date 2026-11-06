"""Locate validated tar headers inside a remote video archive without scanning all metadata."""
import concurrent.futures
import json
import tarfile
from download_holoassist import ROOT, BASE, fetch, inventory


def probe(start):
    block = fetch(BASE + 'video_compress.tar', start, start + 4 * 1024 * 1024 - 1)
    result = []
    for pos in range(0, len(block) - 512, 512):
        if block[pos + 257:pos + 262] != b'ustar':
            continue
        try:
            member = tarfile.TarInfo.frombuf(block[pos:pos + 512], 'utf-8', 'replace')
            if member.isfile() and member.name.endswith('.mp4'):
                result.append({'name': member.name, 'size': member.size, 'offset': start + pos + 512})
        except (tarfile.HeaderError, ValueError):
            pass
    return result


if __name__ == '__main__':
    for batch in range(10):
        starts = [20 * 1024**3 + (batch * 8 + i) * 32 * 1024**2 for i in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            found = [row for rows in pool.map(probe, starts) for row in rows]
        print('Probe batch', batch, found, flush=True)
        if found:
            dest = ROOT / 'video_compress.tar.index.json'
            # Dedicated suffix index retains the original prefix inventory separately.
            if dest.exists() and not (ROOT / 'video_compress.prefix.index.json').exists():
                (ROOT / 'video_compress.prefix.index.json').write_bytes(dest.read_bytes())
            dest.write_text(json.dumps({'members': [found[0]], 'complete': False}, indent=2))
            break
    else:
        raise RuntimeError('No header located within bounded 320 MB probe')
