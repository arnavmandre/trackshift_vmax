/* Pure, dependency-free evidence utilities. Browser global and CommonJS export. */
(function(root){
'use strict';
const copy=x=>JSON.parse(JSON.stringify(x));
const tyres=['FL','FR','RL','RR'];
function validate(input){
 const fail=m=>{throw new Error(m);};
 if(!input||input.schema!=='vmax.predictions.v1')fail('Expected schema vmax.predictions.v1');
 const v=input.video;
 if(!v||typeof v.name!=='string'||!v.name||!Number.isInteger(v.width)||v.width<1||!Number.isInteger(v.height)||v.height<1||!Number.isFinite(v.duration)||v.duration<=0)fail('Video requires name, positive integer width/height and duration in seconds');
 if(!['model','simulated'].includes(input.source))fail('Source must be model or simulated');
 if(typeof input.model_version!=='string'||!input.model_version)fail('model_version is required');
 if(!Array.isArray(input.observations)||input.observations.length>50000)fail('observations must be an array of at most 50000 items');
 const seen=new Set();
 input.observations.forEach((o,i)=>{
  const prefix='Observation '+i+': ';
  if(!o||typeof o.id!=='string'||!o.id||['__proto__','constructor','prototype'].includes(o.id)||seen.has(o.id))fail(prefix+'unique safe string id required');seen.add(o.id);
  if(!Number.isFinite(o.time)||o.time<0||o.time>v.duration)fail(prefix+'time outside video');
  if(o.car_id!==null&&typeof o.car_id!=='string')fail(prefix+'car_id must be string or null');
  if(o.confidence!==null&&(!Number.isFinite(o.confidence)||o.confidence<0||o.confidence>1))fail(prefix+'confidence must be null or 0–1');
  if(!o.points)fail(prefix+'points required');
  for(const k of tyres){const p=o.points[k];if(p!==null&&(!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite)||p[0]<0||p[0]>v.width||p[1]<0||p[1]>v.height))fail(prefix+k+' must be null or an in-image [x,y] pixel point');}
 });
 if(!Array.isArray(input.candidates)||input.candidates.length>5000)fail('candidates must be an array of at most 5000 items');
 const ids=new Set();input.candidates.forEach((c,i)=>{if(!c||typeof c.id!=='string'||!c.id||ids.has(c.id)||!Number.isFinite(c.start)||!Number.isFinite(c.end)||c.start<0||c.end<c.start||c.end>v.duration)fail('Candidate '+i+': unique id and valid start/end seconds required');ids.add(c.id);});
 return copy(input);
}
function nearest(data,t,car){
 if(!data)return null;
 let best=null,delta=Infinity;
 for(const o of data.observations){if(car!==undefined&&o.car_id!==car)continue;const d=Math.abs(o.time-t);if(d<delta){best=o;delta=d;}}
 // Never silently carry predictions across a gap or invent interpolated points.
 return delta<=0.05+1e-8?best:null;
}
function clearance(points,line,flags=[]){
 if(flags.length)return {value:null,reason:'Evidence flagged: '+flags.join(', ')};
 if(!line||line.length!==2)return {value:null,reason:'Draw a local straight boundary first'};
 const [[ax,ay],[bx,by]]=line,dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy);
 if(len<1)return {value:null,reason:'Boundary endpoints must be at least one pixel apart'};
 if(tyres.some(k=>!points?.[k]))return {value:null,reason:'All four contact points are required'};
 const distances=tyres.map(k=>(dx*(points[k][1]-ay)-dy*(points[k][0]-ax))/len);
 return {value:Math.min(...distances),distances,reason:'Signed image-space point distance only; not tyre-footprint clearance or an offence determination'};
}
function demo(video){return validate({schema:'vmax.predictions.v1',source:'simulated',model_version:'DEMO — not measured',video,observations:Array.from({length:8},(_,i)=>({id:'demo-'+i,time:Math.min(video.duration,i*video.duration/8),car_id:'DEMO',confidence:.55+i*.04,points:{FL:[video.width*(.4+i*.012),video.height*.5],FR:[video.width*(.5+i*.012),video.height*.52],RL:[video.width*(.37+i*.012),video.height*.63],RR:[video.width*(.47+i*.012),video.height*.65]}})),candidates:[{id:'demo-event',start:video.duration*.25,end:video.duration*.5}]});}
class History{
 constructor(value){this.value=copy(value);this.undoStack=[];this.redoStack=[];this.audit=[];}
 record(action,before,after){this.audit.push({at:new Date().toISOString(),action,before:copy(before),after:copy(after),source:'human'});}
 set(value,action){const before=this.value;this.undoStack.push(copy(before));this.redoStack=[];this.value=copy(value);this.record(action,before,this.value);}
 undo(){if(!this.undoStack.length)return false;const before=this.value;this.redoStack.push(copy(before));this.value=this.undoStack.pop();this.record('undo',before,this.value);return true;}
 redo(){if(!this.redoStack.length)return false;const before=this.value;this.undoStack.push(copy(before));this.value=this.redoStack.pop();this.record('redo',before,this.value);return true;}
}
const api={validate,nearest,clearance,demo,History,copy,tyres};root.VMAXEvidence=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
