'use strict';
const $=id=>document.getElementById(id), video=$('video'), overlay=$('overlay'), crop=$('crop'), timeline=$('timeline');
let session={clips:[]}, current=null, selectedTrack=1, localURL=null;
const pct=v=>Number.isFinite(v)?`${(100*v).toFixed(1)}%`:'—';
const fmt=v=>Number.isFinite(v)?v.toFixed(3):'—';
function reviewKey(c){return `vmax-pose-review:${session.model_hash||'none'}:${c.video_hash}`;}
function savedReview(c){try{return JSON.parse(localStorage.getItem(reviewKey(c))||'null');}catch{return null;}}
function exportFile(name,data,type){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function queue(){
 const q=$('search').value.toLowerCase(),filter=$('filter').value;$('queue').replaceChildren();
 for(const c of session.clips){
  if(!c.id.toLowerCase().includes(q)||(filter==='candidates'&&!c.candidate_count)||(filter==='pending'&&savedReview(c)))continue;
  const b=document.createElement('button');b.className='clip'+(current?.id===c.id?' active':'');
  const title=document.createElement('div');title.className='id';title.textContent=c.id;
  const sub=document.createElement('small');sub.textContent=c.analysed?`${c.candidate_count} candidates · score ${pct(c.mean_score)}`:'Not analysed';
  b.append(title,sub);b.onclick=()=>select(c);$('queue').append(b);
 }
}
function select(c){
 video.pause();current=c;selectedTrack=c.tracks?.[0]?.id||1;
 $('clipTitle').textContent=c.id;$('clipStatus').textContent=c.analysed?'Saved model predictions':'Local playback · not analysed';
 $('empty').style.display='none';video.src=c.video;video.load();
 $('mean').textContent=pct(c.mean_score);$('coverage').textContent=c.analysed?pct(c.coverage):'—';$('candidates').textContent=c.analysed?c.candidate_count:'—';
 $('track').replaceChildren();for(const t of c.tracks||[]){const o=document.createElement('option');o.value=t.id;o.textContent=`Tracked car ${t.id} · driver unknown`;$('track').append(o);}
 $('track').disabled=!c.tracks?.length;$('prev').disabled=$('next').disabled=!c.fps;
 const prior=c.analysed?savedReview(c):null;$('decision').value=prior?.decision||'unreviewed';$('notes').value=prior?.notes||'';$('saved').textContent='';
 $('save').disabled=$('decision').disabled=$('notes').disabled=!c.analysed;
 $('telemetry').textContent=c.telemetry?`${c.telemetry.samples} telemetry samples. ${c.telemetry.source}.`:'Local video has no linked calibration, telemetry or predictions.';
 refreshEvents();queue();draw();drawTimeline();
}
function frameIndex(){return current?.fps?Math.min(current.frames-1,Math.max(0,Math.floor(video.currentTime*current.fps+1e-4))):null;}
function seekFrame(n){if(!current?.fps)return;video.pause();video.currentTime=Math.max(0,Math.min(current.frames-1,n))/current.fps;}
function refreshEvents(){
 $('events').replaceChildren();let count=0;
 for(const t of current?.tracks||[]){for(const e of t.events){const b=document.createElement('button');b.className='event';b.textContent=`Car ${t.id} · frames ${e.start}–${e.end} · peak +${e.margin.toFixed(2)} m`;b.onclick=()=>{selectedTrack=t.id;$('track').value=t.id;seekFrame(e.peak);draw();};$('events').append(b);count++;}}
 if(!count){const p=document.createElement('p');p.className='muted';p.textContent=current?.analysed?'No candidate windows. This is not a clearance decision.':'No model analysis for this video.';$('events').append(p);}
}
function draw(){
 const W=video.videoWidth||480,H=video.videoHeight||270;overlay.width=W;overlay.height=H;
 const ctx=overlay.getContext('2d');ctx.clearRect(0,0,W,H);
 const f=frameIndex();$('frameLabel').textContent=f===null?`${video.currentTime.toFixed(2)} s · FPS unknown`:`Frame ${f} / ${current.frames-1} · ${video.currentTime.toFixed(2)} s`;
 const cc=crop.getContext('2d');cc.fillStyle='#080b10';cc.fillRect(0,0,crop.width,crop.height);
 $('carInfo').textContent='No observation for this car at this frame.';
 if(!current?.analysed){$('carInfo').textContent='Playback only. No tyre predictions, score or verdict.';drawTimeline();return;}
 if($('showBoundary').checked){ctx.strokeStyle='#7df5c1';ctx.lineWidth=Math.max(1,W/500);ctx.setLineDash([6,5]);for(const line of current.boundary){ctx.beginPath();let active=false;for(const p of line){if(!p||Math.abs(p[0])>W*5||Math.abs(p[1])>H*5){active=false;continue;}if(!active){ctx.moveTo(...p);active=true;}else ctx.lineTo(...p);}ctx.stroke();}ctx.setLineDash([]);}
 for(const t of current.tracks){const row=t.frames.find(r=>r.frame===f);if(!row)continue;
  const active=t.id===Number(selectedTrack),color=row.margin_m>0?'#ffbd69':'#7df5c1';
  if($('showOverlay').checked){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=active?2:1;
   const b=row.box;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
   for(const [i,p] of row.keypoints.entries()){ctx.beginPath();ctx.arc(p[0],p[1],Math.max(2,W/200),0,2*Math.PI);ctx.stroke();ctx.font=`${Math.max(9,W/60)}px sans-serif`;ctx.fillText(['FL','FR','RL','RR'][i],p[0]+4,p[1]-3);}
   ctx.font=`${Math.max(10,W/45)}px sans-serif`;ctx.fillText(`Car ${t.id} · ${pct(row.score)}`,Math.max(0,b[0]),Math.max(14,b[1]-5));
  }
  if(active){$('carInfo').textContent=`Tracked car ${t.id} · driver unknown. Detector score ${pct(row.score)}. Estimated clearance ${row.margin_m>=0?'+':''}${row.margin_m.toFixed(3)} m. Score is uncalibrated.`;
   if(video.readyState>=2){const [x0,y0,x1,y1]=row.box;const x=Math.max(0,x0-10),y=Math.max(0,y0-10),w=Math.min(W-x,x1-x0+20),h=Math.min(H-y,y1-y0+20);if(w>0&&h>0){const s=Math.min(crop.width/w,crop.height/h);cc.drawImage(video,x,y,w,h,(crop.width-w*s)/2,(crop.height-h*s)/2,w*s,h*s);}}
  }
 }
 drawTimeline();
}
function drawTimeline(){
 const ctx=timeline.getContext('2d'),W=timeline.width,H=timeline.height;ctx.clearRect(0,0,W,H);
 if(!current?.analysed)return;
 const vals=current.tracks.flatMap(t=>t.frames.map(f=>f.margin_m)),span=Math.max(1,...vals.map(Math.abs));
 const x=f=>12+(W-24)*f/Math.max(1,current.frames-1),y=m=>H/2-m/span*(H/2-14);
 ctx.strokeStyle='#778496';ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(0,y(0));ctx.lineTo(W,y(0));ctx.stroke();ctx.setLineDash([]);
 for(const t of current.tracks){ctx.strokeStyle=t.id===Number(selectedTrack)?'#7df5c1':'#8298cb';ctx.lineWidth=2;ctx.beginPath();let prev=-2;for(const f of t.frames){if(f.frame!==prev+1)ctx.moveTo(x(f.frame),y(f.margin_m));else ctx.lineTo(x(f.frame),y(f.margin_m));prev=f.frame;}ctx.stroke();}
 ctx.fillStyle='#a4afbd';ctx.font='12px sans-serif';ctx.fillText(`+${span.toFixed(1)} m`,4,12);ctx.fillText(`−${span.toFixed(1)} m`,4,H-3);
 const at=frameIndex();if(at!==null){ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(x(at),0);ctx.lineTo(x(at),H);ctx.stroke();}
}
$('prev').onclick=()=>seekFrame(frameIndex()-1);$('next').onclick=()=>seekFrame(frameIndex()+1);
$('speed').onchange=()=>video.playbackRate=Number($('speed').value);
$('showOverlay').onchange=$('showBoundary').onchange=draw;
$('track').onchange=()=>{selectedTrack=Number($('track').value);draw();};
$('search').oninput=$('filter').onchange=queue;
timeline.onclick=e=>{if(!current?.analysed)return;const rect=timeline.getBoundingClientRect();seekFrame(Math.round((e.clientX-rect.left)/rect.width*(current.frames-1)));};
$('save').onclick=()=>{if(!current?.analysed)return;const record={clip:current.id,video_sha256:current.video_hash,model_sha256:session.model_hash,decision:$('decision').value,notes:$('notes').value,updated_at:new Date().toISOString()};try{localStorage.setItem(reviewKey(current),JSON.stringify(record));$('saved').textContent='Review saved in this browser.';queue();}catch{$('saved').textContent='Browser storage unavailable. Export before closing.';}};
$('export').onclick=()=>{const decisions=session.clips.map(savedReview).filter(Boolean);exportFile('vmax-steward-decisions.json',JSON.stringify({model_sha256:session.model_hash,exported_at:new Date().toISOString(),decisions},null,2),'application/json');};
$('csv').onclick=()=>{const rows=session.metrics?.clips;if(!rows?.length)return;const keys=Object.keys(rows[0]);const cell=x=>'"'+String(x??'').replaceAll('"','""')+'"';exportFile('blind-clip-results.csv',[keys.map(cell).join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\n'),'text/csv');};
$('openLocal').onclick=()=>$('localFile').click();$('localFile').onchange=()=>{const file=$('localFile').files[0];if(!file)return;if(localURL)URL.revokeObjectURL(localURL);localURL=URL.createObjectURL(file);select({id:file.name,video:localURL,analysed:false,tracks:[],fps:null,frames:0});};
for(const event of ['loadeddata','seeked','pause','timeupdate','loadedmetadata'])video.addEventListener(event,draw);
video.addEventListener('error',()=>{$('empty').style.display='grid';$('empty').textContent='Video could not be played. Check the file and browser codec support.';});
if(video.requestVideoFrameCallback){const tick=()=>{draw();video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick);}
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||!current?.fps)return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();seekFrame(frameIndex()+(e.key==='ArrowRight'?1:-1));}});
async function init(){try{const r=await fetch('/api/session');if(!r.ok)throw Error('Session could not be loaded');session=await r.json();$('clipCount').textContent=session.clips.length;$('sessionState').textContent=session.clips.length?`${session.clips.length} clips · saved predictions`:'Training output not installed';
 const metrics=session.metrics?.summary;$('metrics').replaceChildren();if(metrics){const table=document.createElement('table');for(const [label,value] of [['Event precision',pct(metrics.precision)],['Event recall',pct(metrics.recall)],['Matched excursions',metrics.true_positives],['Missed excursions',metrics.missed_events],['False reports',metrics.false_reports],['Median margin error (m)',fmt(metrics.margin_p50_m)],['P95 margin error (m)',fmt(metrics.margin_p95_m)]]){const tr=document.createElement('tr');for(const v of [label,value]){const td=document.createElement('td');td.textContent=v;tr.append(td);}table.append(tr);}$('metrics').append(table);}else $('metrics').textContent='No completed blind-test results available.';
 $('csv').disabled=!session.metrics?.clips?.length;$('provenance').textContent=`${session.model} · selected threshold ${session.threshold??'pending'} · ${session.verification}. Model hash: ${session.model_hash||'pending'}`;
 $('save').disabled=$('decision').disabled=$('notes').disabled=true;$('prev').disabled=$('next').disabled=true;queue();if(session.clips.length)select(session.clips[0]);
 }catch(e){$('sessionState').textContent='Session error';$('empty').textContent=e.message;}}
init();
