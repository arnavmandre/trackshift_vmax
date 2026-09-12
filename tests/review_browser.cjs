// Run against the documented portable sample: node tests/review_browser.cjs.
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');
(async()=>{const server=spawn('python',['-m','vmax_vision.serve','--port','8765','--directory','examples/steward_review']);let browser,page;
try{browser=await chromium.launch({headless:true});page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>{errors.push(e.message);console.error('Page error:',e.stack);});
await page.goto('http://127.0.0.1:8765');await page.locator('video').evaluate(v=>new Promise(resolve=>{if(v.readyState>=2)resolve();else v.addEventListener('loadeddata',resolve,{once:true});}));
assert.ok(await page.locator('video').evaluate(v=>v.videoWidth>0));await page.locator('#next').click();await page.waitForFunction(()=>document.querySelector('#frame').textContent.startsWith('Frame 1 '));
await page.locator('#showOverlay').uncheck();await page.locator('#showOverlay').check();
assert.equal(await page.locator('#count').textContent(),'Not analysed');assert.ok(await page.locator('[data-decision]').first().isDisabled());
assert.deepEqual(errors,[]);console.log('Original-video loading, frame stepping, overlay toggle, and no-results state passed.');
}catch(e){console.error('Browser test failure:',e.message);if(page)console.error(await page.evaluate(()=>({label:document.querySelector('#frame').textContent,time:document.querySelector('video').currentTime,ready:document.querySelector('video').readyState,seekable:Array.from({length:document.querySelector('video').seekable.length},(_,i)=>[document.querySelector('video').seekable.start(i),document.querySelector('video').seekable.end(i)])})));throw e;}finally{if(browser)await browser.close();server.kill();}})().catch(e=>{console.error(e);process.exit(1);});
