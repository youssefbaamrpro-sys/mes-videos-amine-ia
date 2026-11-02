import json, subprocess, numpy as np, wave, httpx, os, re, difflib, unicodedata, sys
SR=48000; TOTAL=182.3667
S=json.load(open("script_fr.json")); B=json.load(open("blocks.json"))
VID=json.load(open("voice2.json"))["voice_id"]
c=httpx.Client(proxy=os.environ.get("HTTPS_PROXY"), timeout=240, verify="/root/.ccr/ca-bundle.crt")
SP=json.load(open("speeds.json"))
texts=[" ".join(S[i][2] for i in b) for b in B]
prev=[]
for k,t in enumerate(texts):
    fn=f"tts4/{k:02d}.mp3"
    if os.path.exists(fn): continue
    body={"text":t,"model_id":"eleven_multilingual_v2",
          "voice_settings":{"stability":0.45,"similarity_boost":0.9,"style":0.25,"use_speaker_boost":True,"speed":SP[k]},
          "previous_text":texts[k-1] if k else None,"next_text":texts[k+1] if k+1<len(texts) else None}
    if prev: body["previous_request_ids"]=prev[-3:]
    r=c.post(f"https://api.elevenlabs.io/v1/text-to-speech/{VID}?output_format=mp3_44100_128",json=body,headers={"xi-api-key":"dummy"})
    if r.status_code!=200: print(k,r.status_code,r.text[:300]); sys.exit(1)
    if r.headers.get("request-id"): prev.append(r.headers["request-id"])
    open(fn,"wb").write(r.content)
def load(k,tempo=1.0):
    af=f"atempo={tempo:.4f}" if tempo!=1 else "anull"
    raw=subprocess.check_output(["ffmpeg","-v","error","-i",f"tts4/{k:02d}.mp3","-af",af,"-f","f32le","-ac","1","-ar",str(SR),"-"])
    return np.frombuffer(raw,np.float32).copy()
def bounds(x):
    env=np.convolve(np.abs(x),np.ones(480)/480,"same")
    idx=np.where(env>0.004)[0]
    a=max(0,idx[0]-int(0.06*SR)); b=min(len(x),idx[-1]+int(0.12*SR)); return a,b
place=[]; buf=np.zeros(int(TOTAL*SR)+SR*5,np.float32); prev_end=0
for k,b in enumerate(B):
    x=load(k); a,e=bounds(x); d=(e-a)/SR
    start=max(S[b[0]][0], prev_end+0.15)
    nxt=S[B[k+1][0]][0] if k+1<len(B) else TOTAL-0.25
    win=nxt-start-0.1; tempo=1.0
    if d>win: tempo=min(1.08,d/win); x=load(k,tempo); a,e=bounds(x)
    y=x[a:e].copy(); n=int(0.02*SR); y[:n]*=np.linspace(0,1,n); y[-n:]*=np.linspace(1,0,n)
    s0=int(start*SR); buf[s0:s0+len(y)]+=y
    place.append([k,start,tempo,a/SR,len(y)/SR]); prev_end=start+len(y)/SR
    print(k,f"start={start:.2f} dur={len(y)/SR:.2f} win={win:.2f} tempo={tempo:.3f} end={prev_end:.2f} next={nxt:.2f}")
buf=buf[:int(TOTAL*SR)]
w=wave.open("voice_fr4.wav","wb");w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
w.writeframes((np.clip(buf,-1,1)*32767).astype(np.int16).tobytes());w.close()
json.dump(place,open("place4.json","w"))
def norm(w):
    w=unicodedata.normalize("NFD",w.lower()); w="".join(ch for ch in w if unicodedata.category(ch)!="Mn"); return re.sub(r"[^a-z0-9]","",w)
tmp="/tmp/clip_whisper.wav"
out=[]
for k,start,tempo,a0,dl in place:
    subprocess.run(["ffmpeg","-v","error","-y","-i",f"tts4/{k:02d}.mp3","-ar","16000","-ac","1",tmp],check=True)
    r=c.post("https://api.groq.com/openai/v1/audio/transcriptions",headers={"Authorization":"Bearer dummy"},
        files={"file":("clip.wav",open(tmp,"rb"),"audio/wav")},
        data={"model":"whisper-large-v3","language":"fr","response_format":"verbose_json","timestamp_granularities[]":"word"})
    W=r.json().get("words",[])
    toks=[(i,w) for i in B[k] for w in S[i][2].split() if norm(w)]
    A=[norm(w) for _,w in toks]; Bn=[norm(x["word"]) for x in W]
    tm=[None]*len(toks)
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,A,Bn,autojunk=False).get_opcodes():
        if tag=="equal":
            for q in range(i2-i1): tm[i1+q]=(W[j1+q]["start"],W[j1+q]["end"])
        elif tag=="replace":
            st=W[j1]["start"];en=W[j2-1]["end"];n=i2-i1
            for q in range(n): tm[i1+q]=(st+(en-st)*q/n,st+(en-st)*(q+1)/n)
    for q in range(len(toks)):
        if tm[q] is None:
            p=tm[q-1][1] if q and tm[q-1] else 0; tm[q]=(p,p+0.25)
    for q,(seg,wd) in enumerate(toks):
        out.append({"seg":seg,"w":wd,"s":round(float(start+tm[q][0]/tempo-a0),3),"e":round(float(start+tm[q][1]/tempo-a0),3)})
json.dump(out,open("words_fr.json","w"),ensure_ascii=False)
print(len(out),out[0],out[-1])
