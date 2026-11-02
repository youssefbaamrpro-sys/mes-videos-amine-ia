import subprocess, numpy as np, json, cv2
W,H=360,640
p=subprocess.Popen(["ffmpeg","-v","error","-i","en.mp4","-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
out=[]
def scan(f):
    r,g,b=[f[...,c].astype(int) for c in range(3)]
    navy=(b>22)&(b<120)&(r<60)&(g<55)&(b-r>15)
    white=(r>222)&(g>222)&(b>222)
    ok=cv2.morphologyEx((navy|white).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((1,11),np.uint8)).astype(bool)
    res=[]
    # sliding search for boxes: rows where a long run of ok exists
    rowrun=np.zeros(H,int); rowx=[None]*H
    for y in range(H):
        row=ok[y].astype(np.int8)
        d=np.diff(np.concatenate([[0],row,[0]]))
        st=np.where(d==1)[0]; en=np.where(d==-1)[0]
        if len(st)==0: continue
        L=en-st; k=L.argmax(); rowrun[y]=L[k]; rowx[y]=(st[k],en[k])
    good=rowrun>=40
    segs=[];cur=[];miss=0
    def close(c1,c2): return abs(c1[0]-c2[0])<=14 and abs(c1[1]-c2[1])<=14
    for y in range(H):
        if good[y] and (not cur or close(rowx[y],rowx[cur[-1]])):
            cur.append(y); miss=0; continue
        if cur:
            miss+=1
            if miss<=3: continue
            segs.append(cur); cur=[]; miss=0
        if good[y]: cur=[y]
    if cur: segs.append(cur)
    # merge fragments of the same box (text rows can break edge continuity)
    merged=[]
    for sg in segs:
        if merged:
            pv=merged[-1]
            a=rowx[pv[len(pv)//2]]; b=rowx[sg[len(sg)//2]]
            inter=max(0,min(a[1],b[1])-max(a[0],b[0])); uni=max(a[1],b[1])-min(a[0],b[0])
            if sg[0]-pv[-1]<=5 and uni>0 and inter/uni>0.6 and (sg[-1]-pv[0])<=50:
                merged[-1]=pv+sg; continue
        merged.append(sg)
    segs=merged
    for sg in segs:
        y0,y1=sg[0],sg[-1]; h=y1-y0+1
        if not (20<=h<=50): continue
        mid=sg[2:-2] if len(sg)>6 else sg
        x0=int(np.median([rowx[t][0] for t in mid])); x1=int(np.median([rowx[t][1] for t in mid]))
        sub_n=navy[y0:y1+1,x0:x1]; sub_w=white[y0:y1+1,x0:x1]
        if sub_w.mean()>0.03 and sub_n.mean()>0.28:
            res.append([x0,y0,x1-x0,h])
    return res
i=0
while True:
    buf=p.stdout.read(W*H*3)
    if len(buf)<W*H*3: break
    out.append(scan(np.frombuffer(buf,np.uint8).reshape(H,W,3))); i+=1
json.dump(out,open("navyboxes.json","w")); print(i)
