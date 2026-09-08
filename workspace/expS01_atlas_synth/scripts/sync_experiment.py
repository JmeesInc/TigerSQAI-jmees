"""Copy only shareable source/config/docs into an existing repository experiment; never copy outputs/data."""
import argparse
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ALLOW={'render':{'.py','.txt','.md'},'registration':{'.py','.md'},'configs':{'.yaml'},'assets':{'.yaml'},'docs':{'.md'},'scripts':{'.py'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();repo=a.repo.resolve()
    if not (repo/'.git').exists():p.error('Target must be an existing Git checkout')
    target=repo/'workspace/expS01_atlas_synth';copied=[]
    for folder,suffixes in ALLOW.items():
        for source in sorted((ROOT/folder).rglob('*')):
            if source.is_symlink() or not source.is_file() or source.suffix not in suffixes or '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT);destination=target/relative
            if not a.dry_run:
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            copied.append(str(relative))
    print('\n'.join(copied));print(f'{len(copied)} shareable files; evidence JSON, outputs, masks and poses are never auto-copied')


if __name__=='__main__':main()
