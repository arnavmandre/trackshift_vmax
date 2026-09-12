/* VMAX v2: extends the existing standalone player without a model dependency. */
(()=>{
'use strict';
const E=VMAXEvidence, histories=new Map();let mode='inspect',selection=null,lineStart=null,drag=null,lastClip=null,lastTick=performance.now();
const style=document.createElement('style');style.textContent=`
.workbench{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:16px;margin-top:16px}.workbench h2{margin:0 0 8px}.wb-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}.wb-row>*{min-width:0}.wb-row select{max-width:100%;padding:6px}.wb-row input[type=number]{width:90px;padding:6px}.wb-row label{display:flex;align-items:center;gap:5px}.wb-status{padding:9px;border-left:3px solid var(--amber);background:var(--amber-dim);white-space:pre-line}.wb-status.demo{border:2px solid var(--amber);font-weight:600}.wb-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0}.wb-metrics div{background:var(--bg);padding:10px}.wb-metrics strong{display:block;font-size:17px;margin:5px 0}.wb-help{color:var(--muted);font-size:11px}.wb-audit{max-height:140px;overflow:auto;white-space:pre-wrap;font:11px monospace}.evidence-overlay{position:absolute;inset:0;width:100%;height:100%;z-index:4;pointer-events:none}.evidence-overlay.edit{pointer-events:auto;cursor:crosshair;touch-action:none}.wb-list{display:flex;flex-wrap:wrap;gap:6px}.wb-error{color:#ff8990;min-height:18px}.wb-banner{position:absolute;bottom:30px;left:12px;z-index:5;background:#332614ef;color:#ffdc88;padding:6px;pointer-events:none;font-weight:600}.wb-row button[aria-pressed=true]{border-color:var(--amber);background:var(--amber-dim)}@media(max-width:650px){.wb-metrics{grid-template-columns:1fr}.wb-row select{width:100%}}`;
document.head.append(style);
const panel=document.createElement('section');panel.className='workbench';panel.setAttribute('aria-label','Evidence workbench');
panel.innerHTML=`<h2>Evidence workbench <span class="tiny">· v2 / model-independent</span></h2>
<p class="wb-help">Original video remains the evidence. Imported scores are uncalibrated detector scores, not probabilities of an offence.</p>
<div class="wb-row"><button id="wbImport">Import predictions JSON</button><button id="wbDemo">Load simulated predictions</button><button id="wbManual">Add manual observation</button><button id="wbTemplate">Download format example</button></div>
<input id="wbFile" type="file" accept="application/json,.json" hidden>
<div id="wbError" class="wb-error" role="alert"></div><div id="wbStatus" class="wb-status">Open a video to begin. No model connected.</div>
<div class="wb-row"><label>Observation <select id="wbObservation" aria-label="Observation"></select></label><button id="wbJump">Jump to observation</button><button id="wbNext">Next unreviewed clip</button></div>
<div class="wb-row"><button id="wbInspect" aria-pressed="true">Inspect</button><button id="wbEdit" aria-pressed="false">Edit contacts</button><button id="wbLine" aria-pressed="false">Draw boundary</button><button id="wbFlip">Flip positive side</button><button id="wbUndo">Undo</button><button id="wbRedo">Redo</button></div>
<div class="wb-row"><label>Place / hide tyre <select id="wbTyre"><option>FL</option><option>FR</option><option>RL</option><option>RR</option></select></label><button id="wbMissing">Mark selected tyre missing</button><label><input id="wbOriginal" type="checkbox" checked> Original points</label><label><input id="wbAdjusted" type="checkbox" checked> Adjusted points</label></div>
<p class="wb-help">Edit contacts: drag a point or click to place the selected tyre. Cyan = original; amber = steward-adjusted. Draw boundary: drag along a local straight segment. Positive side follows the endpoint direction; flip to orient it. Line and edits apply only to the selected observation, never automatically to other frames.</p>
<div class="wb-metrics"><div>Imported detection score<strong id="wbConfidence">—</strong><small>Uncalibrated · not offence probability</small></div><div>Original / adjusted distance<strong id="wbDistance">— / — px</strong><small>Minimum signed contact-point distance</small></div><div>Active review time<strong id="wbTime">0 s</strong><small>Visible-tab, recent-interaction time; approximate</small></div></div>
<div id="wbGate" class="wb-help"></div>
<div class="wb-row" id="wbFlags"><span>Suppress measurement:</span><label><input type="checkbox" value="blur"> Blur</label><label><input type="checkbox" value="occlusion"> Occlusion</label><label><input type="checkbox" value="camera movement"> Camera movement</label><label><input type="checkbox" value="uncertain boundary"> Uncertain boundary</label></div>
<details><summary>Review intervals — manual and imported</summary><div class="wb-row"><label>Start (s) <input id="wbStart" type="number" min="0" step="0.001"></label><label>End (s) <input id="wbEnd" type="number" min="0" step="0.001"></label><button id="wbStartNow">Start here</button><button id="wbEndNow">End here</button><button id="wbInterval">Add manual interval</button></div><div id="wbIntervals" class="wb-list"></div><p class="wb-help">Imported intervals are preserved. Manual intervals are bookmarks for review, not detected offences. Remove and recreate a manual interval to change its range; undo restores it.</p></details>
<details><summary>Audit trail and integration</summary><p class="wb-help">Export decisions includes imported originals, manual corrections, review time and edit history. Local logs are editable and are not tamper-proof. Keep the JSON export with the original video.</p><pre id="wbAudit" class="wb-audit"></pre><button id="wbExportPredictions">Export original prediction JSON</button></details>`;
document.querySelector('.workspace').append(panel);
const canvas=document.createElement('canvas');canvas.className='evidence-overlay';canvas.setAttribute('aria-label','Editable contact-point overlay');$('stage').append(canvas);
const banner=document.createElement('div');banner.className='wb-banner';banner.hidden=true;$('stage').append(banner);
function ev(){const c=current();if(!c)return null;c.evidence??={original:null,manual:[],edits:{},intervals:[],audit:[],activeSeconds:0};return c.evidence;}
function editable(){const x=ev();return {manual:x.manual,edits:x.edits,intervals:x.intervals};}
function hist(){if(!current())return null;if(!histories.has(activeId)){const h=new E.History(editable());h.audit=E.copy(ev().audit||[]);histories.set(activeId,h);}return histories.get(activeId);}
function saveHistory(){const h=hist(),x=ev();Object.assign(x,E.copy(h.value));x.audit=E.copy(h.audit);persist();render();}
function change(action,fn){if(!current())return;const h=hist(),value=E.copy(h.value);fn(value);h.set(value,action);saveHistory();}
function observations(){return [...(ev()?.original?.observations||[]),...(ev()?.manual||[])];}
function selected(){return activeId===lastClip?observations().find(o=>o.id===selection)||null:null;}
function effective(o){return ev()?.edits?.[o.id]||{points:E.copy(o.points),line:null,flags:[]};}
function edit(action,fn){const o=selected();if(!o)return;change(action,v=>{v.edits[o.id]??={points:E.copy(o.points),line:null,flags:[]};fn(v.edits[o.id]);});}
function error(e){$('wbError').textContent=e.message||String(e);}
function download(name,obj){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function modeSet(m){mode=m;lineStart=null;drag=null;cropMode=false;update();for(const [id,val] of [['wbInspect','inspect'],['wbEdit','edit'],['wbLine','line']])$(id).setAttribute('aria-pressed',String(val===m));draw();}
function metadata(){if(!hasVideo())throw new Error('Open a playable local video first.');return {name:current().name,width:video.videoWidth,height:video.videoHeight,duration:video.duration};}
function importPredictions(input){
 const data=E.validate(input),v=metadata();
 if(data.video.name!==v.name||data.video.width!==v.width||data.video.height!==v.height||Math.abs(data.video.duration-v.duration)>.1)throw new Error('Video name, dimensions or duration do not match the selected video. Check your export; no predictions were imported.');
 if(ev().original)throw new Error('This clip already has an original prediction set. Use a separate review session to compare versions; originals cannot be replaced here.');
 if(data.observations.some(o=>ev().manual.some(m=>m.id===o.id)))throw new Error('An imported observation ID collides with a manual observation.');
 ev().original=data;ev().audit.push({at:new Date().toISOString(),action:'import predictions',source:data.source,model_version:data.model_version});histories.delete(activeId);lastClip=activeId;selection=data.observations[0]?.id||null;persist();render();list();toast('Predictions imported. Review the source and uncertainty labels.');return {observations:data.observations.length,candidates:data.candidates.length};
}
// Public integration boundary: takes validated JSON, never runs or trains a model.
window.VMAX={version:'2.0.0',importPredictions,validatePredictions:E.validate,exportReview:()=>E.copy(state)};
$('export').onclick=()=>{readDraft();persist();download('vmax-review-v2.json',{...E.copy(state),application_version:'2.0.0',exportedAt:new Date().toISOString(),modelStatus:'no_live_model',notice:'Human review aid. Imported scores are uncalibrated. Simulated evidence is labelled per clip. Image-space point distances are not metric footprint clearance. Original video bytes are not included; local audit is not tamper-proof.'});};
function render(){
 const x=ev(),o=selected(),s=$('wbObservation');s.replaceChildren();s.add(new Option('Select an observation',''));for(const q of observations())s.add(new Option(`${timeText(q.time)} · ${q.car_id??'Unknown car'} · ${q.id}`,q.id));s.value=selection||'';
 const source=x?.original?.source,sim=source==='simulated';$('wbStatus').classList.toggle('demo',sim);$('wbStatus').textContent=!current()?'Open a video to begin. No model connected.':sim?'SIMULATED OUTPUT — demonstration only. These points do not describe the video.':source==='model'?`Imported predictions · ${x.original.model_version}\nNot independently verified. Metadata matching is not a content-hash identity check.`:'Manual evidence only · no model connected.';
 document.querySelector('.header .badge').textContent=sim?'SIMULATED OUTPUT':source?'Imported predictions':'Models not connected';
 $('stageFoot').textContent=sim?'SIMULATED OVERLAY — not measured':source?'Imported overlay · not independently verified':'Local playback · no model results';
 $('wbConfidence').textContent=o?.confidence==null?'—':`${(o.confidence*100).toFixed(1)}%${sim?' DEMO':''}`;
 const e=o?effective(o):null,a=E.clearance(o?.points,e?.line,e?.flags),b=E.clearance(e?.points,e?.line,e?.flags),fmt=x=>x.value==null?'—':x.value.toFixed(1);
 $('wbDistance').textContent=`${fmt(a)} / ${fmt(b)} px`;$('wbGate').textContent=o?b.reason:'Select or create an observation to inspect evidence.';
 for(const q of $('wbFlags').querySelectorAll('input'))q.checked=e?.flags.includes(q.value)||false;
 $('wbUndo').disabled=!hist()?.undoStack.length;$('wbRedo').disabled=!hist()?.redoStack.length;
 for(const id of ['wbEdit','wbLine','wbFlip','wbMissing','wbJump'])$(id).disabled=!o;
 const intervals=$('wbIntervals');intervals.replaceChildren();for(const [items,origin] of [[x?.original?.candidates||[],sim?'SIMULATED':'Imported'],[x?.intervals||[],'Manual']])for(const c of items){const wrap=document.createElement('span'),b=document.createElement('button');b.textContent=`${origin} ${timeText(c.start)}–${timeText(c.end)}`;b.onclick=()=>seek(c.start);wrap.append(b);if(origin==='Manual'){const r=document.createElement('button');r.textContent='×';r.setAttribute('aria-label','Remove manual interval');r.onclick=()=>change('remove interval',v=>v.intervals=v.intervals.filter(a=>a.id!==c.id));wrap.append(r);}intervals.append(wrap);}
 $('wbAudit').textContent=(x?.audit||[]).slice(-30).map(a=>`${a.at} · ${a.action} · ${a.source||'human'}`).join('\n')||'No edits recorded.';draw();
}
function frameMatches(){return selected()&&hasVideo()&&Math.abs(selected().time-video.currentTime)<=.05&&!video.seeking;}
function draw(){
 const r=$('stage').getBoundingClientRect();canvas.width=Math.round(r.width);canvas.height=Math.round(r.height);const g=canvas.getContext('2d');g.clearRect(0,0,canvas.width,canvas.height);const o=selected(),match=frameMatches();canvas.classList.toggle('edit',mode!=='inspect'&&!!match);banner.hidden=!ev()?.original&&!o;
 banner.textContent=ev()?.original?.source==='simulated'?'SIMULATED OUTPUT — NOT VIDEO MEASUREMENTS':o?'Manual / imported evidence overlay':'';
 if(!o||!match){if(o)banner.textContent+=' · jump to selected observation';return;}
 const vr=videoRect(),sx=vr.width/video.videoWidth,sy=vr.height/video.videoHeight,ox=vr.left-r.left,oy=vr.top-r.top,pt=p=>[ox+p[0]*sx,oy+p[1]*sy],e=effective(o);
 function points(points,color){g.strokeStyle=color;g.fillStyle=color;g.font='bold 12px Arial';for(const k of E.tyres)if(points[k]){const [x,y]=pt(points[k]);g.beginPath();g.arc(x,y,6,0,7);g.stroke();g.fillText(k,x+9,y-7);}}
 if($('wbOriginal').checked)points(o.points,'#55ddff');if($('wbAdjusted').checked&&ev().edits[o.id])points(e.points,'#ffbf55');
 if(e.line){const [a,b]=e.line.map(pt);g.strokeStyle='#ffffff';g.lineWidth=2;g.beginPath();g.moveTo(...a);g.lineTo(...b);g.stroke();g.fillStyle='#fff';g.fillText('A',a[0],a[1]-8);g.fillText('B',b[0],b[1]-8);}
}
function pixel(e){const p=point(e);return [Math.round(p.x*video.videoWidth),Math.round(p.y*video.videoHeight)];}
canvas.onpointerdown=e=>{e.stopPropagation();if(!frameMatches())return;video.pause();const p=pixel(e);if(mode==='line')lineStart=p;else if(mode==='edit'){const pts=effective(selected()).points;let key=$('wbTyre').value,best=20*video.videoWidth/videoRect().width;for(const k of E.tyres)if(pts[k]){const d=Math.hypot(pts[k][0]-p[0],pts[k][1]-p[1]);if(d<best){key=k;best=d;}}drag={key,p};}canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{e.stopPropagation();if(drag)drag.p=pixel(e);};
canvas.onpointerup=e=>{e.stopPropagation();const p=pixel(e);if(lineStart){const a=lineStart;lineStart=null;if(Math.hypot(p[0]-a[0],p[1]-a[1])<1)return;edit('draw boundary',v=>v.line=[a,p]);}else if(drag){const k=drag.key;drag=null;edit('move '+k,v=>v.points[k]=p);}};
canvas.onpointercancel=()=>{drag=null;lineStart=null;};
$('wbImport').onclick=()=>$('wbFile').click();$('wbFile').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>20*1024*1024)throw new Error('Prediction JSON exceeds the 20 MB import limit.');importPredictions(JSON.parse(await f.text()));$('wbError').textContent='';}catch(err){error(err);}finally{e.target.value='';}};
$('wbDemo').onclick=()=>{try{importPredictions(E.demo(metadata()));$('wbError').textContent='';}catch(e){error(e);}};
$('wbTemplate').onclick=()=>download('vmax-predictions-example-SIMULATED.json',E.demo({name:'example.mp4',width:640,height:360,duration:3}));
$('wbManual').onclick=()=>{try{metadata();const id='manual-'+(crypto.randomUUID?.()||Date.now());lastClip=activeId;change('add manual observation',v=>v.manual.push({id,time:video.currentTime,car_id:current().car||null,confidence:null,points:{FL:null,FR:null,RL:null,RR:null}}));selection=id;video.pause();render();modeSet('edit');}catch(e){error(e);}};
$('wbObservation').onchange=e=>{selection=e.target.value;modeSet('inspect');render();};$('wbJump').onclick=()=>{if(selected())seek(selected().time);};
$('wbInspect').onclick=()=>modeSet('inspect');$('wbEdit').onclick=()=>{modeSet('edit');toast('Drag a point, or click to place the selected tyre. Jump to the observation first.');};$('wbLine').onclick=()=>modeSet('line');
$('wbFlip').onclick=()=>edit('flip boundary orientation',v=>{if(v.line)v.line.reverse();});$('wbMissing').onclick=()=>edit('mark '+$('wbTyre').value+' missing',v=>v.points[$('wbTyre').value]=null);
$('wbUndo').onclick=()=>{if(hist()?.undo())saveHistory();};$('wbRedo').onclick=()=>{if(hist()?.redo())saveHistory();};
$('wbOriginal').onchange=$('wbAdjusted').onchange=draw;
for(const q of $('wbFlags').querySelectorAll('input'))q.onchange=()=>edit('evidence quality flags',v=>v.flags=Array.from($('wbFlags').querySelectorAll('input:checked'),q=>q.value));
$('wbStartNow').onclick=()=>$('wbStart').value=(video.currentTime||0).toFixed(3);$('wbEndNow').onclick=()=>$('wbEnd').value=(video.currentTime||0).toFixed(3);
$('wbInterval').onclick=()=>{try{metadata();const start=Number($('wbStart').value),end=Number($('wbEnd').value);if(!$('wbStart').value||!$('wbEnd').value||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start||end>video.duration)throw new Error('Enter start/end within the video, with end ≥ start.');change('add manual interval',v=>v.intervals.push({id:'interval-'+Date.now(),start,end,source:'human'}));$('wbError').textContent='';}catch(e){error(e);}};
$('wbNext').onclick=()=>{const i=state.clips.findIndex(c=>c.id===activeId),all=state.clips.slice(i+1).concat(state.clips.slice(0,i));const c=all.find(c=>!c.review||c.review.decision==='unreviewed');if(c)choose(c.id);else toast('No other unreviewed clips.');};
$('wbExportPredictions').onclick=()=>{if(ev()?.original)download('vmax-original-predictions.json',ev().original);else toast('No imported predictions to export.');};
// Record human assessments without changing original model evidence.
const saveReview=$('saveReview').onclick;$('saveReview').onclick=()=>{const before=E.copy(current()?.review||null);saveReview();if(current()){ev().audit.push({at:new Date().toISOString(),action:'save human review',source:'human',before,after:E.copy(current().review)});histories.delete(activeId);persist();render();}};
const oldSummary=summary;summary=()=>oldSummary().replace('Model status: not connected; no analysis','Live model: not connected; imported predictions may be present').replace('Clearance / detector confidence: unavailable','World-space clearance: unavailable; imported detector scores are uncalibrated')+ '\nEvidence source: '+(ev()?.original?.source||'manual only')+'\nImported model: '+(ev()?.original?.model_version||'none')+'\nManual edits: '+Object.keys(ev()?.edits||{}).length+'\nSee JSON export for original predictions, corrections and limitations.';
const cropClick=$('selectCrop').onclick;$('selectCrop').onclick=()=>{modeSet('inspect');cropClick();};
let interaction=performance.now();for(const name of ['pointerdown','keydown'])document.addEventListener(name,()=>interaction=performance.now());
setInterval(()=>{const now=performance.now(),dt=Math.min(2,(now-lastTick)/1000);lastTick=now;if(current()&&!document.hidden&&now-interaction<60000){ev().activeSeconds+=dt;}$('wbTime').textContent=Math.round(ev()?.activeSeconds||0)+' s';if(activeId!==lastClip){lastClip=activeId;selection=null;modeSet('inspect');render();}},500);
setInterval(()=>{if(current())persist();},10000);
for(const name of ['timeupdate','seeked','loadeddata','pause'])video.addEventListener(name,draw);window.addEventListener('resize',draw);
document.addEventListener('keydown',e=>{if(document.querySelector('dialog[open]')||e.target.closest('input,textarea,select'))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$(e.shiftKey?'wbRedo':'wbUndo').click();}if(e.key.toLowerCase()==='n'&&!e.ctrlKey&&!e.metaKey)$('wbNext').click();});
$('helpDialog').querySelector('h2').textContent='VMAX · Steward Prototype v2';
render();
})();
