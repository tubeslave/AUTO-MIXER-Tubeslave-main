import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { basename } from "node:path";

const execFileAsync = promisify(execFile);

export type TrackAnalysis = {
  file:string; path:string;
  technical:{durationSec:number|null;sampleRate:number|null;channels:number|null;codec:string|null;bitDepth:number|null};
  signal:{hasSignal:boolean;peakDbfs:number|null;rmsDbfs:number|null;crestFactor:number|null;clippingSuspected:boolean;silenceRatio:number|null;silenceRegions:Array<{start:number;end:number;duration:number}>};
  loudness:{integratedLUFS:number|null;loudnessRangeLU:number|null;truePeakDbTP:number|null;momentary:Array<{t:number;lufs:number}>;shortTerm:Array<{t:number;lufs:number}>};
  spectral:{centroidHz:number|null;rolloffHz:number|null;flatness:number|null;flux:number|null;slope:number|null;crest:number|null};
  source:{guess:string;family:string;confidence:number;alternatives:string[];method:"heuristic"};
  bleed:{status:"not_detected"|"suspected"|"uncertain";confidence:number;reason:string};
  warnings:Array<{severity:"info"|"warning"|"error";code:string;message:string}>;
};

function num(v:any){const x=Number(v);return Number.isFinite(x)?x:null}
function last(text:string,re:RegExp){const a=[...text.matchAll(re)];return a.length?num(a[a.length-1][1]):null}
async function ff(args:string[],max=64*1024*1024){return await execFileAsync("ffmpeg",args,{maxBuffer:max}).catch((e:any)=>({stderr:e.stderr||"",stdout:e.stdout||""}))}

export async function ensureFfmpeg(){await execFileAsync("ffmpeg",["-version"]);await execFileAsync("ffprobe",["-version"])}

async function probe(file:string){
  const {stdout}=await execFileAsync("ffprobe",["-v","error","-select_streams","a:0","-show_entries","stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample,duration","-show_entries","format=duration","-of","json",file],{maxBuffer:4*1024*1024});
  const j=JSON.parse(stdout),s=j.streams?.[0]??{};
  return {durationSec:num(s.duration??j.format?.duration),sampleRate:num(s.sample_rate),channels:num(s.channels),codec:s.codec_name??null,bitDepth:num(s.bits_per_raw_sample||s.bits_per_sample)};
}

async function astats(file:string){
  const {stderr}=await ff(["-hide_banner","-nostats","-i",file,"-af","astats=metadata=0:reset=0","-f","null","-"],16*1024*1024);
  return {peakDbfs:last(stderr,/Peak level dB:\s*([-\d.]+)/g),rmsDbfs:last(stderr,/RMS level dB:\s*([-\d.]+)/g),crestFactor:last(stderr,/Crest factor:\s*([-\d.]+)/g)};
}

async function silence(file:string,duration:number|null){
  const {stderr}=await ff(["-hide_banner","-nostats","-i",file,"-af","silencedetect=noise=-60dB:d=0.25","-f","null","-"],16*1024*1024);
  const starts=[...stderr.matchAll(/silence_start:\s*([0-9.]+)/g)].map(m=>Number(m[1]));
  const ends=[...stderr.matchAll(/silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)/g)].map(m=>({end:Number(m[1]),duration:Number(m[2])}));
  const regions=[] as Array<{start:number;end:number;duration:number}>;
  for(let i=0;i<Math.min(starts.length,ends.length);i++)regions.push({start:starts[i],end:ends[i].end,duration:ends[i].duration});
  const total=regions.reduce((a,b)=>a+b.duration,0);
  return {regions,silenceRatio:duration&&duration>0?Math.max(0,Math.min(1,total/duration)):null};
}

async function loudness(file:string){
  const {stderr}=await ff(["-hide_banner","-loglevel","verbose","-nostats","-i",file,"-af","ebur128=peak=true:framelog=verbose","-f","null","-"]);
  const momentary=[] as Array<{t:number;lufs:number}>,shortTerm=[] as Array<{t:number;lufs:number}>;
  for(const m of stderr.matchAll(/t:\s*([0-9.]+).*?M:\s*([-\d.]+).*?S:\s*([-\d.]+)/g)){
    const t=Number(m[1]),M=Number(m[2]),S=Number(m[3]);if(Number.isFinite(M)&&M>-100)momentary.push({t,lufs:M});if(Number.isFinite(S)&&S>-100)shortTerm.push({t,lufs:S});
  }
  const thin=(a:any[])=>a.length<=600?a:a.filter((_,i)=>i%Math.ceil(a.length/600)===0);
  return {integratedLUFS:last(stderr,/\bI:\s*([-\d.]+)\s*LUFS/g),loudnessRangeLU:last(stderr,/\bLRA:\s*([-\d.]+)\s*LU/g),truePeakDbTP:last(stderr,/\bPeak:\s*([-\d.]+)\s*dBFS/g),momentary:thin(momentary),shortTerm:thin(shortTerm)};
}

async function spectral(file:string){
  const {stderr}=await ff(["-hide_banner","-nostats","-i",file,"-af","silenceremove=start_periods=1:start_duration=0.05:start_threshold=-55dB:stop_periods=-1:stop_duration=0.15:stop_threshold=-55dB,aspectralstats=measure=centroid+rolloff+flatness+flux+slope+crest,ametadata=print","-f","null","-"]);
  const vals=(k:string)=>[...stderr.matchAll(new RegExp(`lavfi\\.aspectralstats\\.\\d+\\.${k}=([-+0-9.eE]+)`,`g`))].map(m=>Number(m[1])).filter(Number.isFinite);
  const avg=(a:number[])=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
  return {centroidHz:avg(vals("centroid")),rolloffHz:avg(vals("rolloff")),flatness:avg(vals("flatness")),flux:avg(vals("flux")),slope:avg(vals("slope")),crest:avg(vals("crest"))};
}

function classify(file:string,centroid:number|null,crest:number|null){
  const n=basename(file).toLowerCase();let guess="unknown",family="unknown",confidence=.15;
  const set=(g:string,f:string,c:number)=>{guess=g;family=f;confidence=c};
  if(/vox|vocal|voc|вокал/.test(n))set("vocal","vocals",.96);else if(/bass|бас/.test(n))set("bass","bass",.96);else if(/gtr|guitar|гитар/.test(n))set("guitar","guitars",.94);else if(/kick|боч/.test(n))set("kick","drums",.96);else if(/snare|мал|рабоч/.test(n))set("snare","drums",.96);else if(/overhead|oh|оверх/.test(n))set("overheads","drums",.94);else if(centroid!=null&&centroid<900&&crest!=null&&crest>4)set("low-frequency percussive/bass source","unknown",.43);else if(centroid!=null&&centroid>3500&&crest!=null&&crest>3)set("bright/percussive source","unknown",.40);
  return {guess,family,confidence,alternatives:[],method:"heuristic" as const};
}

export async function analyzeTrack(file:string):Promise<TrackAnalysis>{
  const technical=await probe(file);const [a,s,l,sp]=await Promise.all([astats(file),silence(file,technical.durationSec),loudness(file),spectral(file)]);
  const hasSignal=a.rmsDbfs!=null&&a.rmsDbfs>-90;const clippingSuspected=(a.peakDbfs!=null&&a.peakDbfs>=-.01)||(l.truePeakDbTP!=null&&l.truePeakDbTP>=0);
  const signal={hasSignal,peakDbfs:a.peakDbfs,rmsDbfs:a.rmsDbfs,crestFactor:a.crestFactor,clippingSuspected,silenceRatio:s.silenceRatio,silenceRegions:s.regions};
  const source=classify(file,sp.centroidHz,a.crestFactor);
  const bleed=hasSignal?{status:"uncertain" as const,confidence:.2,reason:"Для надёжного bleed-анализа требуется междорожечное сравнение."}:{status:"not_detected" as const,confidence:.7,reason:"Сигнал отсутствует."};
  const warnings:TrackAnalysis["warnings"]=[];if(!hasSignal)warnings.push({severity:"error",code:"NO_SIGNAL",message:"Аудиосигнал практически отсутствует."});if(clippingSuspected)warnings.push({severity:"error",code:"CLIPPING",message:"Обнаружен или вероятен цифровой/True Peak clipping."});
  return {file:basename(file),path:file,technical,signal,loudness:l,spectral:sp,source,bleed,warnings};
}
