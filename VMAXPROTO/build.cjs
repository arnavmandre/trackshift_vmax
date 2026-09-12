const fs=require('fs'),path=require('path');
const dir=__dirname,base=fs.readFileSync(path.join(dir,'base.html'),'utf8');
const scripts=['src/evidence.js','src/workbench.js'].map(f=>'<script>\n'+fs.readFileSync(path.join(dir,f),'utf8')+'\n</script>').join('\n');
fs.writeFileSync(path.join(dir,'VMAX-Steward-Review.html'),base.replace('</body>',scripts+'\n</body>'));
console.log('Built standalone VMAX-Steward-Review.html');
