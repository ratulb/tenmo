const { marked } = require('marked');
const fs = require('fs');
const path = require('path');

const rootDir = fs.realpathSync(__dirname);
const docsDir = path.join(rootDir, 'md');
const outDir = path.join(rootDir, 'generated');

console.log('rootDir:', rootDir);
console.log('docsDir:', docsDir);
console.log('outDir:', outDir);

function fixExternalLinks(html) {
  return html.replace(/href="\/mojo\/std\//g, 'href="https://docs.modular.com/mojo/stdlib/');
}

const template = (title, content) => `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${title} - Tenmo</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    code { background: #f4f4f4; padding: 2px 6px; }
    pre { background: #f4f4f4; padding: 16px; overflow-x: auto; }
    a { color: #ff5500; }
  </style>
</head>
<body>
  <div style="margin-bottom: 20px; border-bottom: 1px solid #eee;">
    <a href="/tenmo/">Home</a> | <a href="/tenmo/tensor/index.html">Tensor</a> | <a href="/tenmo/shapes/index.html">Shapes</a>
  </div>
  ${content}
</body>
</html>`;

function stripFrontmatter(md) {
  if (md.startsWith('---')) {
    const end = md.indexOf('---', 3);
    if (end !== -1) return md.substring(end + 3);
  }
  return md;
}

function findMdFiles(dir) {
  const files = [];
  for (const f of fs.readdirSync(dir)) {
    const fullPath = path.join(dir, f);
    if (fs.statSync(fullPath).isDirectory()) files.push(...findMdFiles(fullPath));
    else if (f.endsWith('.md')) files.push(fullPath);
  }
  return files;
}

fs.mkdirSync(outDir, { recursive: true });

const files = findMdFiles(docsDir);
console.log('Found', files.length, 'md files');

for (const fullPath of files) {
  const file = path.basename(fullPath);
  let md = fs.readFileSync(fullPath, 'utf8');
  md = stripFrontmatter(md);
  let html = marked(md);
  html = fixExternalLinks(html);
  const title = md.match(/^title:\s*(.+)$/m)?.[1] || file.replace('.md', '');
  const out = template(title, html);
  
  const outFile = path.join(outDir, file.replace('.md', '.html'));
  fs.writeFileSync(outFile, out);
  console.log('Generated:', file);
}

console.log('Done! ' + files.length + ' files in ' + outDir);