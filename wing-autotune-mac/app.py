import json, math, os, socket, subprocess, threading, time, webbrowser, tkinter as tk
from tkinter import messagebox
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from email.parser import BytesParser
from email.policy import default
import numpy as np
import sounddevice as sd
from scipy import signal
from pythonosc.udp_client import SimpleUDPClient

APP='WING AutoTune'; DATA=Path.home()/'Library'/'Application Support'/APP; DATA.mkdir(parents=True,exist_ok=True); CFG=DATA/'config.json'
DEFAULT={'audio':{'input_device':None,'output_device':None,'sample_rate':48000,'input_channel':1,'level_dbfs':-20.0},'measurement':{'signal':'pink','seconds':8.0,'calibration_file':None},'wing':{'ip':'192.168.1.120','port':2223},'systems':{'left':{'name':'LEFT','kind':'matrix','target':1,'output_channels':[1],'band':'main'},'right':{'name':'RIGHT','kind':'matrix','target':2,'output_channels':[2],'band':'main'},'sub1':{'name':'SUB 1','kind':'matrix','target':3,'output_channels':[3],'band':'sub'},'sub2':{'name':'SUB 2','kind':'matrix','target':4,'output_channels':[4],'band':'sub'}},'safety':{'min_coherence':0.75,'max_cut_db':6.0},'server':{'port':8765}}
def merge(a,b):
    if isinstance(a,dict):
        o=dict(b) if isinstance(b,dict) else {}
        for k,v in a.items(): o[k]=merge(v,o.get(k))
        return o
    return a if b is None else b
def load():
    c=merge(DEFAULT,json.loads(CFG.read_text()) if CFG.exists() else {}); CFG.write_text(json.dumps(c,indent=2,ensure_ascii=False)); return c
def save(c): CFG.write_text(json.dumps(c,indent=2,ensure_ascii=False))
def lip():
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:s.connect(('8.8.8.8',80));return s.getsockname()[0]
    except:return '127.0.0.1'
    finally:s.close()
def speak(t):
    try:subprocess.Popen(['say',t],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except:pass
def pink(n,sr):
    r=np.random.default_rng();f=np.fft.rfftfreq(n,1/sr);p=r.uniform(0,2*np.pi,len(f));a=np.ones_like(f);a[1:]=1/np.sqrt(np.maximum(f[1:],1));y=np.fft.irfft(a*np.exp(1j*p),n);return (y/max(np.max(np.abs(y)),1e-9)).astype(np.float32)
def sweep(n,sr):
    t=np.arange(n)/sr;T=n/sr;f0,f1=20,20000;K=T/math.log(f1/f0);L=f0*K;return np.sin(2*np.pi*L*(np.exp(t/K)-1)).astype(np.float32)
def calread(p):
    if not p or not Path(p).exists():return None
    z=[]
    for line in Path(p).read_text(errors='ignore').splitlines():
        line=line.strip()
        if not line or line[0] in '#;*':continue
        q=line.replace(',','.').split()
        if len(q)>1:
            try:z.append((float(q[0]),float(q[1])))
            except:pass
    return np.array(sorted(z)) if len(z)>1 else None
def calc(cal,f):
    if cal is None:return np.zeros_like(f)
    return np.interp(np.log10(np.maximum(f,1e-6)),np.log10(cal[:,0]),cal[:,1],left=cal[0,1],right=cal[-1,1])
def gcc(r,m,sr):
    n=1<<int(np.ceil(np.log2(len(r)+len(m))));R=np.fft.rfft(r,n);M=np.fft.rfft(m,n);G=M*np.conj(R);G/=np.maximum(np.abs(G),1e-12);c=np.fft.irfft(G,n);mx=min(int(sr*.25),n//2-1);c=np.r_[c[-mx:],c[:mx+1]];sh=int(np.argmax(np.abs(c))-mx);return sh/sr,float(c[sh+mx])
def tf(r,m,sr,cal):
    d,pk=gcc(r,m,sr);sh=int(round(d*sr))
    if 0<sh<len(r)-2048:r,m=r[:-sh],m[sh:]
    elif sh<0 and -sh<len(r)-2048:k=-sh;r,m=r[k:],m[:-k]
    nper=min(16384,max(2048,len(r)//2));f,Pxx=signal.welch(r,fs=sr,nperseg=nper);_,Pyy=signal.welch(m,fs=sr,nperseg=nper);_,Pxy=signal.csd(m,r,fs=sr,nperseg=nper);co=np.clip(abs(Pxy)**2/np.maximum(Pxx*Pyy,1e-30),0,1);H=Pxy/np.maximum(Pxx,1e-30);mag=20*np.log10(np.maximum(abs(H),1e-12))+calc(cal,f);ph=((np.unwrap(np.angle(H))*180/np.pi+180)%360)-180;q=(f>=20)&(f<=20000);return f[q],mag[q],co[q],ph[q],d*1000,pk
def peq(f,mag,co,c,band):
    lo,hi=(35,180) if band=='sub' else (45,12000);v=(f>=lo)&(f<=hi)&(co>=c['safety']['min_coherence']);res=mag.copy();o=[]
    for _ in range(4):
        s=res.copy();s[~v]=-999;i=int(np.argmax(s))
        if s[i]<1:break
        g=-min(float(s[i]),c['safety']['max_cut_db']);o.append({'freq':round(float(f[i]),1),'gain':round(g,2),'q':1.2 if band=='sub' else 1.5});res-=s[i]*np.exp(-.5*(np.log2(np.maximum(f,1)/f[i])/.45)**2)
    return o
class Engine:
    def __init__(self):self.cfg=load();self.results={};self.corr={};self.busy=False;self.msg='Готов'
    def devices(self):return [{'id':i,'name':d['name'],'inputs':d['max_input_channels'],'outputs':d['max_output_channels']} for i,d in enumerate(sd.query_devices())]
    def capture(self,outs):
        c=self.cfg;sr=int(c['audio']['sample_rate']);n=int(sr*float(c['measurement']['seconds']));r=(pink(n,sr) if c['measurement']['signal']=='pink' else sweep(n,sr))*10**(float(c['audio']['level_dbfs'])/20);ii=c['audio']['input_device'];oo=c['audio']['output_device']
        if ii is None or oo is None:raise RuntimeError('Выберите аудиоустройства')
        inch=int(c['audio']['input_channel'])-1;ocs=[int(x)-1 for x in outs];nch=max(ocs)+1;rec=np.zeros(n,np.float32);ptr=0
        def cb(indata,outdata,frames,ti,status):
            nonlocal ptr
            e=min(ptr+frames,n);outdata[:]=0;ch=r[ptr:e]
            for x in ocs:outdata[:len(ch),x]=ch
            rec[ptr:e]=indata[:len(ch),inch];ptr=e
            if ptr>=n:raise sd.CallbackStop
        with sd.Stream(device=(ii,oo),samplerate=sr,channels=(sd.query_devices(ii)['max_input_channels'],nch),dtype='float32',blocksize=1024,callback=cb):
            while ptr<n:sd.sleep(50)
        return r,rec
    def measure(self,k):
        s=self.cfg['systems'][k];self.busy=True;self.msg='Измеряем '+s['name'];speak(self.msg);r,m=self.capture(s['output_channels']);f,mag,co,ph,d,pk=tf(r,m,self.cfg['audio']['sample_rate'],calread(self.cfg['measurement'].get('calibration_file')));mask=((f>=35)&(f<=140)) if s['band']=='sub' else ((f>=80)&(f<=12000));md=float(np.nanmedian(co[mask])) if np.any(mask) else 0;eq=peq(f,mag,co,self.cfg,s['band']);self.results[k]={'name':s['name'],'delay_ms':round(d,3),'coherence':round(md,3),'polarity_suggest':bool(pk<-.2 and md>.75),'freqs':f.tolist(),'mag':mag.tolist(),'coh':co.tolist(),'phase':ph.tolist(),'eq':eq};latest=max(x['delay_ms'] for x in self.results.values());self.corr={x:{'delay_ms':round(max(0,latest-y['delay_ms']),3),'delay_m':round(max(0,latest-y['delay_ms'])*.343,3),'eq':y['eq'],'invert':y['polarity_suggest']} for x,y in self.results.items()};self.busy=False;self.msg='Готово';return self.results[k]
    def apply(self,k,opt):
        s=self.cfg['systems'][k];c=self.corr[k];u=SimpleUDPClient(self.cfg['wing']['ip'],int(self.cfg['wing']['port']));root='/mtx' if s['kind']=='matrix' else '/main';b=f"{root}/{s['target']}"
        if opt.get('delay',True):
            dm=float(c['delay_m']);u.send_message(b+'/dly/on',1 if dm>=.1 else 0)
            if dm>=.1:u.send_message(b+'/dly/m',min(100,max(.1,dm)))
        if opt.get('polarity',False):u.send_message(b+'/in/set/inv',1 if c['invert'] else 0)
        if opt.get('eq',True):
            e=b+'/eq';u.send_message(e+'/mdl','STD');u.send_message(e+'/on',1);u.send_message(e+'/mix',100.0)
            for i in range(1,5):
                if i<=len(c['eq']):q=c['eq'][i-1];u.send_message(f'{e}/{i}f',q['freq']);u.send_message(f'{e}/{i}g',q['gain']);u.send_message(f'{e}/{i}q',q['q'])
                else:u.send_message(f'{e}/{i}g',0.0)
E=Engine()
HTML='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:-apple-system;background:#07100f;color:#f4f7f7;margin:0}.a{max-width:430px;margin:auto;padding:18px}.b{font-size:28px;font-weight:800}.b i{color:#0a84ff;font-style:normal}.c{background:#121d1c;border:1px solid #283532;border-radius:17px;padding:14px;margin:12px 0}.g{display:grid;grid-template-columns:1fr 1fr;gap:10px}.s{background:#0d1716;border:1px solid #283532;border-radius:15px;padding:11px}.m{font-size:12px;display:flex;justify-content:space-between;padding:5px 0}.x{width:100%;border:0;border-radius:12px;padding:11px;margin-top:8px;background:#0a84ff;color:white;font-weight:800}.d{background:#26322f}.f{width:100%;padding:9px;background:#09110f;color:white;border:1px solid #33413e;border-radius:9px;margin:4px 0 9px}.sm{font-size:11px;color:#8fa19d}canvas{width:100%;height:140px;background:#09110f;border-radius:10px}</style><div class=a><div class=b>WING <i>AutoTune</i></div><div class=sm id=st>...</div><div class=c><b>Фаза • Coherence • Delay</b><div class=sm>Измеряйте LEFT, RIGHT, SUB 1, SUB 2 по отдельности из одной точки FOH.</div></div><div class=g id=sys></div><div class=c><label><input id=eq type=checkbox checked> PEQ</label> <label><input id=del type=checkbox checked> Delay</label> <label><input id=pol type=checkbox> Polarity</label></div><div class=c><b>Фаза</b><canvas id=ph></canvas><b>Когерентность</b><canvas id=co></canvas></div><div class=c><b>Настройки</b><input class=f id=ip placeholder="WING IP"><select class=f id=ind></select><select class=f id=out></select><input class=f id=mic type=number placeholder="Mic channel"><input class=f id=lvl type=number placeholder="dBFS"><select class=f id=sig><option value=pink>Pink noise</option><option value=sweep>ESS sweep</option></select><input class=f id=cal type=file><button class="x d" onclick=up()>Загрузить calibration</button><div id=maps></div><button class=x onclick=save()>Сохранить настройки</button></div></div><script>let S={};const $=x=>document.querySelector(x);async function A(u,o){let r=await fetch(u,o),j=await r.json();if(!r.ok)throw Error(j.detail||'error');return j}async function R(){S=await A('/state');$('#st').textContent=`Mac ${S.ip} • ${S.msg}`;draw();fill()}function draw(){let K=['left','right','sub1','sub2'];$('#sys').innerHTML=K.map(k=>{let s=S.cfg.systems[k],r=S.results[k],c=S.corr[k];return `<div class=s><b>${s.name}</b><div class=sm>${s.kind.toUpperCase()} ${s.target} • OUT ${s.output_channels}</div>${r?`<div class=m><span>Arrival</span><b>${r.delay_ms} ms</b></div><div class=m><span>Coherence</span><b>${r.coherence}</b></div><div class=m><span>Add delay</span><b>${c.delay_ms} ms</b></div><div class=m><span>Polarity</span><b>${r.polarity_suggest?'check Ø':'OK'}</b></div>`:''}<button class=x onclick="M('${k}')">${r?'Повторить':'Измерить'}</button>${r?`<button class='x d' onclick="P('${k}')">В WING</button>`:''}</div>`}).join('');L($('#ph'),K.map(k=>S.results[k]?.phase),-180,180);L($('#co'),K.map(k=>S.results[k]?.coh),0,1)}function L(c,a,mx,mn){let x=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight,d=devicePixelRatio;c.width=w*d;c.height=h*d;x.scale(d,d);x.clearRect(0,0,w,h);let C=['#2ee77d','#0a84ff','#ffd15a','#ff5d58'];a.forEach((v,j)=>{if(!v)return;x.strokeStyle=C[j];x.beginPath();v.forEach((y,i)=>{let X=i*w/(v.length-1),Y=(mn-y)/(mn-mx)*h;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke()})}async function M(k){try{await A('/measure?key='+k,{method:'POST'});R()}catch(e){alert(e)}}async function P(k){await A('/apply?key='+k,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({eq:$('#eq').checked,delay:$('#del').checked,polarity:$('#pol').checked})});alert('Отправлено')}async function D(){let z=await A('/devices');z.forEach(q=>{if(q.inputs)$('#ind').add(new Option(`${q.id}: ${q.name}`,q.id));if(q.outputs)$('#out').add(new Option(`${q.id}: ${q.name}`,q.id))})}function fill(){if(!S.cfg)return;$('#ip').value=S.cfg.wing.ip;$('#mic').value=S.cfg.audio.input_channel;$('#lvl').value=S.cfg.audio.level_dbfs;$('#sig').value=S.cfg.measurement.signal;if(S.cfg.audio.input_device!==null)$('#ind').value=S.cfg.audio.input_device;if(S.cfg.audio.output_device!==null)$('#out').value=S.cfg.audio.output_device;$('#maps').innerHTML=['left','right','sub1','sub2'].map(k=>{let s=S.cfg.systems[k];return `<b>${s.name}</b><input class=f id='map_${k}' value='${s.kind},${s.target},${s.output_channels.join(',')}'>`}).join('')}async function save(){let p={wing:{ip:$('#ip').value},audio:{input_device:+$('#ind').value,output_device:+$('#out').value,input_channel:+$('#mic').value,level_dbfs:+$('#lvl').value},measurement:{signal:$('#sig').value},systems:{}};for(let k of ['left','right','sub1','sub2']){let q=$('#map_'+k).value.split(',');p.systems[k]={kind:q[0],target:+q[1],output_channels:q.slice(2).map(Number)}}await A('/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});R()}async function up(){let f=$('#cal').files[0];if(!f)return;let z=new FormData();z.append('file',f);await fetch('/cal',{method:'POST',body:z});alert('Calibration загружен')}R();D();setInterval(R,1800)</script>'''
class H(BaseHTTPRequestHandler):
    def _send(self,obj,code=200,typ='application/json'):
        b=obj.encode() if isinstance(obj,str) else json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type',typ+'; charset=utf-8');self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path)
        if p.path=='/':return self._send(HTML,typ='text/html')
        if p.path=='/state':return self._send({'ip':lip(),'msg':E.msg,'cfg':E.cfg,'results':E.results,'corr':E.corr})
        if p.path=='/devices':return self._send(E.devices())
        self._send({'error':'not found'},404)
    def do_POST(self):
        p=urlparse(self.path);ln=int(self.headers.get('Content-Length','0'));raw=self.rfile.read(ln)
        try:
            if p.path=='/measure':return self._send(E.measure(parse_qs(p.query)['key'][0]))
            if p.path=='/apply':E.apply(parse_qs(p.query)['key'][0],json.loads(raw or b'{}'));return self._send({'ok':True})
            if p.path=='/config':E.cfg=merge(E.cfg,json.loads(raw));save(E.cfg);return self._send({'ok':True})
            if p.path=='/cal':
                msg=BytesParser(policy=default).parsebytes(b'Content-Type: '+self.headers['Content-Type'].encode()+b'\r\nMIME-Version: 1.0\r\n\r\n'+raw);part=next(x for x in msg.iter_parts() if x.get_filename());dest=DATA/'mic.cal';dest.write_bytes(part.get_payload(decode=True));E.cfg['measurement']['calibration_file']=str(dest);save(E.cfg);return self._send({'ok':True})
        except Exception as e:return self._send({'detail':str(e)},500)
        self._send({'error':'not found'},404)
    def log_message(self,*a):pass
srv=None
def start_server():
    global srv
    if srv:return
    srv=ThreadingHTTPServer(('0.0.0.0',int(E.cfg['server']['port'])),H);threading.Thread(target=srv.serve_forever,daemon=True).start()
root=tk.Tk();root.title(APP);root.geometry('470x285');root.resizable(False,False);root.configure(bg='#0b1211');st=tk.StringVar(value='Запуск…')
def openui():start_server();webbrowser.open(f"http://127.0.0.1:{E.cfg['server']['port']}");st.set(f"Работает\nMac: http://127.0.0.1:{E.cfg['server']['port']}\nСмартфон: http://{lip()}:{E.cfg['server']['port']}")
def phone():messagebox.showinfo('Смартфон',f"Откройте в той же сети:\nhttp://{lip()}:{E.cfg['server']['port']}")
tk.Label(root,text='WING AutoTune',font=('Helvetica Neue',28,'bold'),fg='white',bg='#0b1211').pack(pady=(24,4));tk.Label(root,text='PA calibration • phase • coherence • delay',font=('Helvetica Neue',12),fg='#84a09b',bg='#0b1211').pack();tk.Label(root,textvariable=st,font=('Helvetica Neue',12),fg='#d4dedc',bg='#0b1211',justify='center').pack(pady=18);f=tk.Frame(root,bg='#0b1211');f.pack();tk.Button(f,text='Открыть интерфейс',command=openui,width=18,height=2).grid(row=0,column=0,padx=5);tk.Button(f,text='Адрес смартфона',command=phone,width=18,height=2).grid(row=0,column=1,padx=5);root.after(700,openui);root.mainloop()
