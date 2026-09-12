const {test}=require('node:test'),assert=require('node:assert/strict'),E=require('../src/evidence.js');
const video={name:'test.mp4',width:640,height:360,duration:3};
test('demo explicitly labels simulated evidence',()=>{const d=E.demo(video);assert.equal(d.source,'simulated');assert.equal(d.observations.length,8);});
test('validator copies rather than aliases originals',()=>{const d=E.demo(video),v=E.validate(d);v.observations[0].points.FL[0]=0;assert.notEqual(d.observations[0].points.FL[0],0);});
for(const [name,mutate] of [
 ['schema',d=>d.schema='wrong'],['duplicate observation',d=>d.observations.push(d.observations[0])],['NaN score',d=>d.observations[0].confidence=NaN],['score over one',d=>d.observations[0].confidence=2],['out of image',d=>d.observations[0].points.FL=[641,0]],['missing tyre key',d=>delete d.observations[0].points.RR],['negative time',d=>d.observations[0].time=-1],['reversed interval',d=>d.candidates[0].start=3],['duplicate interval',d=>d.candidates.push(d.candidates[0])],['missing car identity',d=>delete d.observations[0].car_id]
])test('reject '+name,()=>{const d=E.demo(video);mutate(d);assert.throws(()=>E.validate(d));});
test('unknown tyre and score stay null',()=>{const d=E.demo(video);d.observations[0].confidence=null;d.observations[0].points.FL=null;assert.equal(E.validate(d).observations[0].points.FL,null);});
test('reserved object-key IDs are rejected',()=>{const d=E.demo(video);d.observations[0].id='__proto__';assert.throws(()=>E.validate(d));});
test('does not propagate observations across gaps',()=>{const d=E.demo(video);assert.equal(E.nearest(d,.1),null);assert.equal(E.nearest(d,0).id,'demo-0');});
test('car filter does not associate other car',()=>{assert.equal(E.nearest(E.demo(video),0,'other'),null);});
const points={FL:[0,2],FR:[2,3],RL:[0,4],RR:[2,5]},line=[[0,0],[10,0]];
test('signed pixel distances and orientation',()=>{assert.equal(E.clearance(points,line).value,2);assert.equal(E.clearance(points,[...line].reverse()).value,-5);});
test('missing point suppresses measurement',()=>assert.equal(E.clearance({...points,RR:null},line).value,null));
test('quality flag suppresses measurement',()=>assert.equal(E.clearance(points,line,['blur']).value,null));
test('degenerate boundary suppresses measurement',()=>assert.equal(E.clearance(points,[[0,0],[0,0]]).value,null));
test('undo and redo append audit without destroying original history',()=>{const h=new E.History({x:1});h.set({x:2},'move');h.undo();assert.equal(h.value.x,1);h.redo();assert.equal(h.value.x,2);assert.deepEqual(h.audit.map(a=>a.action),['move','undo','redo']);h.undo();h.set({x:3},'new');assert.equal(h.redo(),false);});
