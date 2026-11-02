import json, subprocess, numpy as np, wave, httpx, os, time, sys
sys.path.insert(0,"/home/user/mes-videos-amine-ia/.claude/skills/amine")
from clients import telecharger
SR=48000; MODEL=sys.argv[1] if len(sys.argv)>1 else "sync/lipsync-2-pro"
shots=json.load(open("shots.json"))
c=httpx.Client(proxy=os.environ.get("HTTPS_PROXY"), timeout=300, verify="/root/.ccr/ca-bundle.crt")
H={"Authorization":"Bearer dummy"}
raw=subprocess.check_output(["ffmpeg","-v","error","-i","dub_final.wav","-f","s16le","-ac","1","-ar",str(SR),"-"])
A=np.frombuffer(raw,np.int16)
def up(p,ct):
    r=c.post("https://api.replicate.com/v1/files",headers=H,files={"content":(os.path.basename(p),open(p,"rb"),ct)}); r.raise_for_status(); return r.json()["urls"]["get"]
preds={}
for k,(s,e) in enumerate(shots):
    out=f"shots/ls_{k:02d}.mp4"
    if os.path.exists(out): continue
    v=f"shots/v_{k:02d}.mp4"; a=f"shots/a_{k:02d}.wav"
    subprocess.run(["ffmpeg","-v","error","-y","-i","video_fr_noaudio.mp4","-vf",f"select='between(n\\,{s}\\,{e-1})',setpts=N/30/TB","-r","30","-an","-c:v","libx264","-crf","15",v],check=True)
    y=A[int(s/30*SR):int(s/30*SR)+int((e-s)/30*SR)]
    w=wave.open(a,"wb");w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR);w.writeframes(y.tobytes());w.close()
    body={"input":{"video":up(v,"video/mp4"),"audio":up(a,"audio/wav"),"sync_mode":"cut_off","temperature":0.5}}
    for t in range(6):
        r=c.post(f"https://api.replicate.com/v1/models/{MODEL}/predictions",headers=H,json=body)
        if r.status_code==429: time.sleep(12); continue
        break
    d=r.json()
    if r.status_code>=300: print(k,r.status_code,d); sys.exit(1)
    preds[k]=d["id"]; print(k,"started",d["id"],flush=True); time.sleep(11)
while preds:
    time.sleep(10)
    for k,pid in list(preds.items()):
        d=c.get(f"https://api.replicate.com/v1/predictions/{pid}",headers=H).json()
        if d["status"]=="succeeded":
            telecharger(d["output"],f"shots/ls_{k:02d}.mp4"); print(k,"ok",flush=True); del preds[k]
        elif d["status"] in ("failed","canceled"):
            print(k,"FAILED",d.get("error")); del preds[k]
