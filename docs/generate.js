const { marked } = require('marked');
const fs = require('fs');
const path = require('path');

const docsDir = path.join(__dirname, 'md');
const outDir = path.join(__dirname, 'md', 'build');

// Convert /mojo/std/... to docs.modular.com/mojo/stdlib/...
function fixExternalLinks(html) {
  return html
    .replace(/href="\/mojo\/std\//g, 'href="https://docs.modular.com/mojo/stdlib/');
}

// Simple HTML template
const template = (title, content) => `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - Tenmo</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
    pre { background: #f4f4f4; padding: 16px; border-radius: 6px; overflow-x: auto; }
    h1, h2, h3 { color: #333; }
    a { color: #ff5500; }
    .nav { display: flex; gap: 20px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
    .toc { position: fixed; right: 20px; top: 100px; font-size: 14px; }
    @media (max-width: 900px) { .toc { display: none; } }
  </style>
</head>
<body>
  <div class="nav">
    <a href="/tenmo/">Home</a>
    <a href="/tenmo/tensor.html">Tensor</a>
    <a href="/tenmo/shapes.html">Shapes</a>
  </div>
  ${content}
</body>
</html>`;

// Strip YAML frontmatter
function stripFrontmatter(md) {
  if (md.startsWith('---')) {
    const end = md.indexOf('---', 3);
    if (end !== -1) {
      return md.substring(end + 3);
    }
  }
  return md;
}

fs.mkdirSync(outDir, { recursive: true });

const files = fs.readdirSync(docsDir).filter(f => f.endsWith('.md'));

for (const file of files) {
  let md = fs.readFileSync(path.join(docsDir, file), 'utf8');
  md = stripFrontmatter(md);
  let html = marked(md);
  html = fixExternalLinks(html);
  const title = md.match(/^title:\s*(.+)$/m)?.[1] || file.replace('.md', '');
  const out = template(title, html);
  fs.writeFileSync(path.join(outDir, file.replace('.md', '.html')), out);
  console.log(`Generated: ${file.replace('.md', '.html')}`);
}

console.log(`\nDone! ${files.length} files generated in ${outDir}/`);