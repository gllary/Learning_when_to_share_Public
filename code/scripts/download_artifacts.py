"""Download the published model and data archives into the repository."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'gllary/Learning_when_to_share_Public'
TAG = 'v1.0.0'
PREFIX = 'hpsafemoe-artifacts.tar.gz.part-'

class ChunkReader(io.RawIOBase):
    """Expose ordered archive parts as one continuous byte stream."""
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None
    def readable(self):
        return True
    def readinto(self, buffer):
        view = memoryview(buffer)
        count = 0
        while count < len(view):
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            n = self.current.readinto(view[count:])
            if n:
                count += n
            else:
                self.current.close()
                self.current = None
        return count
    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()

def download(asset, directory):
    target = directory / asset['name']
    if target.is_file() and target.stat().st_size == asset['size']:
        return target
    partial = target.with_suffix(target.suffix + '.download')
    for attempt in range(3):
        try:
            request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'HP-SafeMoE'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            if partial.stat().st_size != asset['size']:
                raise OSError(f'Incomplete download: {target.name}')
            partial.replace(target)
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('Download failed')

def extract(paths, destination):
    count = 0
    destination = destination.resolve()
    with io.BufferedReader(ChunkReader(paths)) as stream, tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] not in {'data', 'checkpoint', 'records', 'results'}:
                raise ValueError(f'Invalid artifact path: {member.name}')
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Artifact escapes the repository: {member.name}')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not (member.isfile() or member.islnk()):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            if member.islnk():
                link = Path(member.linkname)
                if link.is_absolute() or '..' in link.parts:
                    raise ValueError(f'Invalid artifact link: {member.linkname}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if member.isfile() and target.stat().st_size != member.size:
                    raise FileExistsError(f'Remove the incomplete local file before retrying: {target}')
                continue
            if member.islnk():
                linked = (destination / member.linkname).resolve()
                if not linked.is_relative_to(destination):
                    raise ValueError(f'Artifact link escapes the repository: {member.linkname}')
                source = linked.open('rb')
            else:
                source = archive.extractfile(member)
            if source is None:
                raise ValueError(f'Cannot read artifact: {member.name}')
            partial = target.with_name(target.name + '.partial_download')
            try:
                with source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-downloads', action='store_true', help='Retain the downloaded archive parts after extraction')
    args = parser.parse_args()
    request = urllib.request.Request(f'https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}', headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'HP-SafeMoE'})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    assets = sorted((a for a in release['assets'] if a['name'].startswith(PREFIX)), key=lambda a: a['name'])
    if not assets or [a['name'] for a in assets] != [f'{PREFIX}{i:03d}' for i in range(len(assets))]:
        raise ValueError('The release archive is missing or has nonconsecutive parts')
    cache = ROOT / 'runs/artifact_download'
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, asset in enumerate(assets, 1):
        print(f'Downloading part {index}/{len(assets)}: {asset["name"]}', flush=True)
        paths.append(download(asset, cache))
    print('Restoring models and data...', flush=True)
    count = extract(paths, ROOT)
    if not args.keep_downloads:
        for path in paths:
            path.unlink()
        cache.rmdir()
    print(f'Complete: restored {count} files in {ROOT}')

if __name__ == '__main__':
    main()
