import sys, numpy as np, subprocess
sys.argv=[sys.argv[0]]
import importlib.util
spec=importlib.util.spec_from_file_location("r","03_rendu.py"); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
from PIL import Image
ts=[float(x) for x in open("ts.txt").read().split()]
ims=[]
for t in ts:
    i=int(round(t*30))
    raw=subprocess.check_output(["ffmpeg","-v","error","-i","en.mp4","-vf",f"select=eq(n\\,{i})","-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","-"])
    f=np.frombuffer(raw,np.uint8).reshape(640,360,3)
    ims.append(r.process(f,i).resize((360,640)))
cols=8; rows=(len(ims)+cols-1)//cols
sheet=Image.new("RGB",(360*cols,640*rows))
for k,im in enumerate(ims): sheet.paste(im,((k%cols)*360,(k//cols)*640))
sheet.save("test_sheet.jpg",quality=85)
