"""Colorize already-rendered uint8 labels for QA; never used to create labels."""
import argparse
import colorsys
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='label PNG or output directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--palette',type=Path,help='YAML mapping fine_id -> [R,G,B], e.g. local official colors')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--mapping',default='assets/class_mapping.yaml')
    args=p.parse_args()
    palette=np.zeros((31,3),dtype=np.uint8)
    for i in range(1,31):palette[i]=np.array(colorsys.hsv_to_rgb((i*0.61803398875)%1,.65,.95))*255
    if args.palette:
        values=yaml.safe_load(args.palette.read_text())
        values=values.get('palette',values)
        for i,color in values.items():palette[int(i)]=color
    names={0:'Background'}|{c['fine_id']:c['name'] for c in yaml.safe_load(Path(args.mapping).read_text())['classes']}
    files=[args.input] if args.input.is_file() else sorted((args.input/'label').glob('*.png'))[:args.limit]
    if not files:raise ValueError('No label PNGs')
    args.output.mkdir(parents=True,exist_ok=True)
    previews=[]
    for path in files:
        image=Image.open(path)
        if image.mode!='L':raise ValueError('Expected uint8 single-channel L label PNG')
        labels=np.array(image)
        if labels.max()>30:raise ValueError('Label out of range')
        out=Image.fromarray(palette[labels]);out.save(args.output/(path.stem+'_color.png'))
        scale=min(512/out.width,288/out.height)
        thumb=out.resize((round(out.width*scale),round(out.height*scale)),Image.Resampling.NEAREST)
        tile=Image.new('RGB',(512,320));tile.paste(thumb,(0,24))
        ImageDraw.Draw(tile).text((6,4),path.stem,fill='white');previews.append(tile)
    columns=min(3,len(previews));rows=(len(previews)+columns-1)//columns
    sheet=Image.new('RGB',(columns*512,rows*320+31*18),(25,25,25))
    for i,tile in enumerate(previews):sheet.paste(tile,((i%columns)*512,(i//columns)*320))
    draw=ImageDraw.Draw(sheet)
    for i in range(31):
        y=rows*320+i*18
        draw.rectangle((8,y+2,23,y+15),fill=tuple(palette[i]))
        draw.text((30,y),f'{i}: {names[i]}',fill='white')
    sheet.save(args.output/'contact_sheet.png')
    (args.output/'palette.json').write_text(json.dumps({str(i):palette[i].tolist() for i in range(31)},indent=2))
    print(args.output/'contact_sheet.png')


if __name__=='__main__':main()
