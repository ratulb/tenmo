const { marked } = require('marked');
const fs = require('fs');
const path = require('path');

const rootDir = __dirname;
const docsDir = path.join(rootDir, 'docs');
const outDir = path.join(rootDir, 'generated');

console.log('rootDir:', rootDir);
console.log('docsDir:', docsDir);
console.log('outDir:', outDir);

function fixExternalLinks(html) {
  html = html.replace(/href="\/mojo\/std\//g, 'href="https://docs.modular.com/mojo/stdlib/');
  html = html.replace(/href="\/mojo\/tenmo\//g, '');
  return html;
}

function getNavHtml(currentFile) {
  return `<div class="nav">
    <a href="index.html">Home</a>
    <a href="Tensor.html">Tensor</a>
    <a href="Shape.html">Shapes</a>
  </div>`;
}

const template = (title, content) => `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${title} - Tenmo</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
    pre { background: #f4f4f4; padding: 16px; border-radius: 6px; overflow-x: auto; }
    h1, h2, h3 { color: #333; }
    a { color: #ff5500; }
    .nav { display: flex; gap: 20px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
  </style>
</head>
<body>
  ${getNavHtml()}
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

let indexContent = '<h1>Tenmo Documentation</h1><ul>';

for (const fullPath of files) {
  const file = path.basename(fullPath);
  let md = fs.readFileSync(fullPath, 'utf8');
  md = stripFrontmatter(md);
  let html = marked(md);
  html = fixExternalLinks(html);
  const titleMatch = md.match(/^title:\s*(.+)$/m);
  const title = titleMatch ? titleMatch[1] : file.replace('.md', '');
  const out = template(title, html);
  
  const outFile = path.join(outDir, file.replace('.md', '.html'));
  fs.writeFileSync(outFile, out);
  console.log('Generated:', file);
  indexContent += `<li><a href="${file.replace('.md', '.html')}">${title}</a></li>`;
}

indexContent += '</ul>';
const indexOut = template('Tenmo', indexContent);
fs.writeFileSync(path.join(outDir, 'index.html'), indexOut);

console.log('Done! ' + files.length + ' files in ' + outDir);