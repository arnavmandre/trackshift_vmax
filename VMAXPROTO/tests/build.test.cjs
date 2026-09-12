const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../VMAX-Steward-Review.html'),'utf8');
test('all embedded scripts parse',()=>{const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.equal(scripts.length,3);for(const [,source] of scripts)new vm.Script(source);});
test('no external runtime scripts or stylesheets',()=>{assert.doesNotMatch(html,/<script[^>]+src=/i);assert.doesNotMatch(html,/<link[^>]+stylesheet/i);});
test('all literal UI element references have unique IDs',()=>{const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);assert.equal(ids.length,new Set(ids).size);const references=[...html.matchAll(/\$\('([^']+)'\)/g)].map(x=>x[1]);assert.deepEqual([...new Set(references)].filter(x=>!ids.includes(x)),[]);});
test('visible simulation and measurement limitations exist',()=>{assert.match(html,/SIMULATED OUTPUT — NOT VIDEO MEASUREMENTS/);assert.match(html,/not tyre-footprint clearance/);assert.match(html,/not tamper-proof/);});
