import subprocess, numpy as np, json, cv2, sys
W,H=360,640
src=sys.argv[1] if len(sys.argv)>1 else "src.mp4"
def scan(f):
    r,g,b=[f[...,c].astype(np.int16) for c in range(3)]
    white=((r>232)&(g>232)&(b>232)).astype(np.float32)
    black=(r<110)&(g<110)&(b<110)
    wd=cv2.blur(white,(9,9))
    ink=(black&(wd>0.35)).astype(np.uint8)
    lines=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(13,5)))
    n,lab,st,_=cv2.connectedComponentsWithStats(lines,8)
    res=[]
    for j in range(1,n):
        x,y,w,h,a=st[j]
        if not(5<=h<=28 and w>=10): continue
        dens=ink[y:y+h,x:x+w].mean()
        if dens<0.08: continue
        # must be inside a white box: pixels just outside text (pad 4) mostly white
        X0,Y0,X1,Y1=max(0,x-5),max(0,y-4),min(W,x+w+5),min(H,y+h+4)
        ring=np.concatenate([white[Y0,X0:X1],white[Y1-1,X0:X1],white[Y0:Y1,X0],white[Y0:Y1,X1-1]])
        if ring.mean()<0.45: continue
        res.append([int(x),int(y),int(w),int(h)])
    return res
if __name__=="__main__":
    p=subprocess.Popen(["ffmpeg","-v","error","-i",src,"-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
    out=[]
    while True:
        buf=p.stdout.read(W*H*3)
        if len(buf)<W*H*3: break
        out.append(scan(np.frombuffer(buf,np.uint8).reshape(H,W,3)))
    json.dump(out,open("wboxes.json","w")); print(len(out))
