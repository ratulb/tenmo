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

function getNavHtml() {
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - Tenmo</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <style>
    :root {
      --color-primary: #ff5500;
      --color-bg: #ffffff;
      --color-text: #24292e;
      --color-code-bg: #f6f8fa;
      --color-border: #e1e4e8;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --color-bg: #0d1117;
        --color-text: #c9d1d9;
        --color-code-bg: #161b22;
        --color-border: #30363d;
      }
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px 40px;
      line-height: 1.6;
      color: var(--color-text);
      background: var(--color-bg);
    }
    .nav {
      display: flex;
      gap: 20px;
      margin-bottom: 30px;
      padding-bottom: 15px;
      border-bottom: 1px solid var(--color-border);
    }
    .nav a {
      color: var(--color-primary);
      text-decoration: none;
      font-weight: 500;
    }
    .nav a:hover { text-decoration: underline; }
    code {
      background: var(--color-code-bg);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.9em;
    }
    pre {
      background: var(--color-code-bg);
      padding: 16px;
      border-radius: 6px;
      overflow-x: auto;
      border: 1px solid var(--color-border);
    }
    pre code { background: transparent; padding: 0; }
    h1, h2, h3 { color: var(--color-text); margin-top: 30px; }
    h1 { font-size: 2em; border-bottom: 1px solid var(--color-border); padding-bottom: 10px; }
    a { color: var(--color-primary); }
    .mojo-function-sig {
      background: var(--color-code-bg);
      padding: 12px 16px;
      border-radius: 6px;
      border-left: 3px solid var(--color-primary);
      margin: 16px 0;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 14px;
    }
    .mojo-function-detail {
      margin: 24px 0;
      padding: 16px;
      border-left: 3px solid var(--color-primary);
      background: linear-gradient(to right, var(--color-code-bg), transparent);
    }
    .mojo-alias-header { display: flex; align-items: center; gap: 12px; margin: 20px 0 8px 0; }
    .mojo-alias-sig {
      background: var(--color-code-bg);
      padding: 8px 12px;
      border-radius: 4px;
      font-family: monospace;
      display: inline-block;
      margin: 8px 0;
    }
    .mojo-module-detail { margin: 20px 0; }
    section.mojo-docs { margin-top: 20px; }
    .mojo-docs h2 { font-size: 1.5em; border-bottom: 1px solid var(--color-border); padding-bottom: 8px; }
    .mojo-docs h3 { font-size: 1.2em; }
    .mojo-docs ul { padding-left: 20px; }
    .mojo-docs li { margin: 8px 0; }
    .stability-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
    }
    .stability-stable { background: #28a745; color: white; }
    .stability-experimental { background: #ffc107; color: #333; }
    .stability-deprecated { background: #dc3545; color: white; }
    blockquote {
      border-left: 4px solid var(--color-primary);
      margin: 16px 0;
      padding: 8px 16px;
      background: var(--color-code-bg);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0;
    }
    th, td {
      border: 1px solid var(--color-border);
      padding: 8px 12px;
      text-align: left;
    }
    th { background: var(--color-code-bg); }
  </style>
</head>
<body>
  ${getNavHtml()}
  <section class="mojo-docs">
  ${content}
  </section>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function() {
      document.querySelectorAll('pre code').forEach(function(block) {
        hljs.highlightElement(block);
      });
      document.querySelectorAll('pre').forEach(function(pre) {
        pre.setAttribute('tabindex', '0');
      });
    });
  </script>
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

let indexContent = '<h1>Tenmo API Reference</h1>\n<p>A lean tensor library and neural network framework built in Mojo.</p>\n\n<h2>All Documentation</h2>\n<ul>';

for (const fullPath of files) {
  const file = path.basename(fullPath);
  if (file === 'index.md') continue;
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
  indexContent += `<li><a href="${file.replace('.md', '.html')}">${title}</a></li>\n`;
}

indexContent += '</ul>';
const indexOut = template('Tenmo', indexContent);
fs.writeFileSync(path.join(outDir, 'index.html'), indexOut);

console.log('Done! ' + files.length + ' files in ' + outDir);