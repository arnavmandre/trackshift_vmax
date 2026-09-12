const {chromium}=require('playwright');
const {spawn,execFileSync}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
(async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vmax-ui-'));
 execFileSync('python',['review_fixture.py',dir]);
 const server=spawn('python',['review_server.py','--data',dir,'--port','8877']);
 let browser;
 try{
  for(let i=0;i<100;i++){try{if((await fetch('http://127.0.0.1:8877/api/session')).ok)break;}catch{}await new Promise(r=>setTimeout(r,100));}
  browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8877');await page.waitForFunction(()=>document.querySelector('#video').readyState>=2);
  await page.waitForFunction(()=>document.querySelector('#clipTitle').textContent==='fixture');
  await page.locator('#next').click();await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.startsWith('Frame 1 /'));
  await page.locator('.event').first().click();await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.startsWith('Frame 5 /'));
  await page.locator('#showOverlay').uncheck();await page.locator('#showOverlay').check();
  await page.locator('#decision').selectOption('insufficient');await page.locator('#notes').fill('Browser test: insufficient evidence.');await page.locator('#save').click();
  await page.reload();await page.waitForFunction(()=>document.querySelector('#decision').value==='insufficient');
  if(await page.locator('#notes').inputValue()!=='Browser test: insufficient evidence.')throw Error('Review not persisted');
  const download=page.waitForEvent('download');await page.locator('#export').click();const file=await download;const payload=JSON.parse(fs.readFileSync(await file.path(),'utf8'));if(payload.decisions.length!==1)throw Error('Export missing review');
  await page.screenshot({path:'review-ui-check.png',fullPage:true});
  await page.locator('#localFile').setInputFiles(path.join(dir,'fresh_blind/test.mp4'));
  await page.waitForFunction(()=>document.querySelector('#clipStatus').textContent.includes('not analysed'));
  if(!(await page.locator('#save').isDisabled())||!(await page.locator('#next').isDisabled()))throw Error('Unanalysed local video enables evidence decisions or assumed FPS');
  if(errors.length)throw Error(errors.join('\n'));
  console.log('Browser checks passed: original video, frame seek, overlays, review persistence/export and honest local-video state.');
 }finally{if(browser)await browser.close();server.kill();fs.rmSync(dir,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
