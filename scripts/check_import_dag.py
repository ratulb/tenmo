#!/usr/bin/env python3
"""Check the tenmo library's internal import graph is a DAG.

Parses `from tenmo...` / relative `from .X...` imports in every `.mojo` file
under a root directory (default `tenmo/`), builds the module-level dependency
graph, and reports any strongly-connected component (import cycle). Stdlib-only
(Tarjan SCC, no networkx). `from . import submodule` resolves to the submodule
(`package.submodule`), not to the package root.

Usage:
    python3 scripts/check_import_dag.py                # check tenmo/ is acyclic
    python3 scripts/check_import_dag.py --root tenmo/
    python3 scripts/check_import_dag.py --rootset tenmo.shared
        # additionally require that no module INSIDE the rootset imports
        # anything OUTSIDE it (the redesign's "Layer 0: zero tenmo imports" rule
        # is the special case where the rootset imports nothing at all)

Exit codes: 0 = DAG ok, 1 = cycles (and/or rootset violations) found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_IMPORT_RE = re.compile(
    r'^(?P<kind>from|import)\s+'
    r'(?P<target>[\w.]+)'           # tenmo.foo.bar  |  .foo.bar  |  .
    r'(?:\s+import\s+(?P<names>[\w,\s]+))?'  # names for `from .shared import mnemonics`
)

_SKIP_ROOTS = frozenset({'std', 'python', 'sys', 'time', 'os', 'random'})
_SELF = '.'


def module_name_and_package_for(path: Path, root: Path, rootname: str):
    """Return (module_name, package_name) fully-qualified under `rootname`.

    `tenmo/__init__.mojo`      -> (tenmo, tenmo)
    `tenmo/tensor.mojo`        -> (tenmo.tensor, tenmo)
    `tenmo/kernels/foo.mojo`   -> (tenmo.kernels.foo, tenmo.kernels)
    `tenmo/shared/__init__.mojo` -> (tenmo.shared, tenmo.shared)
    """
    rel = path.relative_to(root)
    parts = [rootname] + list(rel.parts)
    assert parts[-1].endswith('.mojo')
    stem = parts[-1][:-len('.mojo')]
    dirs = parts[:-1]
    if stem == '__init__':
        module = '.'.join(dirs)          # the package itself
        package = module
    else:
        module = '.'.join(dirs + [stem])
        package = '.'.join(dirs)
    return module, package


def iter_imports(text: str):
    """Yield (kind, target) for each logical import statement in `text`."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not (stripped.startswith('from ') or stripped.startswith('import ')):
            i += 1
            continue
        depth = stripped.count('(') - stripped.count(')')
        buf = stripped
        while depth > 0 and i + 1 < len(lines):
            i += 1
            nxt = lines[i].strip()
            buf += ' ' + nxt
            depth += nxt.count('(') - nxt.count(')')
        m = _IMPORT_RE.match(buf)
        if m:
            yield m.group('kind'), m.group('target'), m.group('names')
        i += 1


def resolve(target: str, package: str, names: str | None = None) -> str | None:
    """Resolve an import target to a tenmo module name, or None if external.

    For `from .shared import mnemonics` the target is `.`; the imported submodule
    name resolves to `package.<name>` rather than the package root itself.
    """
    if target == _SELF:
        if names:
            return f'{package}.{names.split(",")[0].strip()}'
        return package
    if target.startswith('.'):
        dots = len(target) - len(target.lstrip('.'))
        rel_parts = [p for p in target.lstrip('.').split('.') if p]
        up = dots - 1
        pkg_parts = package.split('.') if package else []
        for _ in range(up):
            if pkg_parts:
                pkg_parts = pkg_parts[:-1]
        return '.'.join(pkg_parts + rel_parts)
    root = target.split('.')[0]
    if root in _SKIP_ROOTS:
        return None
    if root != 'tenmo':
        return None
    return target


def tarjan_scc(graph: dict[str, set[str]]):
    """Return list of SCCs (each a set of node names) with size > 1."""
    index = {}
    lowlink = {}
    on_stack = set()
    stack = []
    counter = [0]
    result = []

    def strongconnect(v: str) -> None:
        index[v] = lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            if len(comp) > 1 or (len(comp) == 1 and v in graph.get(v, ())):
                result.append(comp)

    for node in graph:
        if node not in index:
            strongconnect(node)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='tenmo',
                        help='root directory of .mojo sources (default: tenmo)')
    parser.add_argument('--rootset', action='append', default=[],
                        help='module prefix whose members must not import '
                             'outside the prefix (Layer-0 purity check); '
                             'repeatable')
    parser.add_argument('--rootset-imports', action='append', default=[],
                        help='allow a rootset to import from another prefix: '
                             'FORMAT <rootset>:<allowed-prefix> (e.g. '
                             'tenmo.gpu:tenmo.shared); repeatable')
    args = parser.parse_args()

    root = Path(args.root)
    rootname = root.name
    files = sorted(root.rglob('*.mojo'))

    modules: dict[str, Path] = {}
    packages: dict[str, str] = {}
    for f in files:
        mod, pkg = module_name_and_package_for(f, root, rootname)
        modules[mod] = f
        packages[mod] = pkg

    graph: dict[str, set[str]] = {m: set() for m in modules}
    missing: list[str] = []
    for f in files:
        mod, _ = module_name_and_package_for(f, root, rootname)
        text = f.read_text(errors='replace')
        for _kind, target, names in iter_imports(text):
            resolved = resolve(target, packages[mod], names)
            if resolved is None:
                continue
            if resolved == mod:
                continue
            if resolved not in modules:
                missing.append(f'{mod} -> {resolved}')
                continue
            graph[mod].add(resolved)

    cycles = tarjan_scc(graph)
    problems = 0

    if missing:
        print(f'[warn] imports of unknown tenmo modules ({len(missing)}):')
        for line in missing[:10]:
            print(f'  {line}')
        print('  (more omitted...)') if len(missing) > 10 else None

    print(f'modules: {len(modules)}  internal edges: '
          f'{sum(len(v) for v in graph.values())}')

    if cycles:
        print(f'\nERROR: {len(cycles)} import cycle(s) found:')
        for comp in sorted(cycles, key=lambda c: sorted(c)[0]):
            members = sorted(comp)
            label = ', '.join(members)
            if len(members) > 12:
                label = ', '.join(members[:12]) + f', ... (+{len(members)-12} more)'
            print(f'  SCC ({len(members)} modules): {label}')
            edges = sorted(f'{v} -> {w}'
                           for v in members for w in graph.get(v, ()) if w in comp)
            shown = edges[:15]
            for edge in shown:
                print(f'    {edge}')
            if len(edges) > 15:
                print(f'    ... (+{len(edges)-15} more edges within SCC)')
        problems += 1

    if args.rootset:
        allowed_by_rootset: dict[str, list[str]] = {}
        for spec in args.rootset_imports:
            rs, sep, dep = spec.partition(':')
            if not sep:
                print(f'[warn] --rootset-imports needs <rootset>:<prefix>, '
                      f'got {spec!r}')
                continue
            allowed_by_rootset.setdefault(rs, []).append(dep)
        for prefix in args.rootset:
            allowed = allowed_by_rootset.get(prefix, [])
            outside = []
            for modname in graph:
                if modname == prefix or modname.startswith(prefix + '.'):
                    for dep in graph[modname]:
                        inside = (
                            dep == prefix or dep.startswith(prefix + '.')
                            or any(
                                dep == a or dep.startswith(a + '.')
                                for a in allowed
                            )
                        )
                        if not inside:
                            outside.append(f'{modname} -> {dep}')
            if outside:
                print(f'\nERROR: rootset `{prefix}` imports outside itself '
                      f'({len(outside)} edge(s)):')
                for line in sorted(outside):
                    print(f'  {line}')
                problems += 1
            else:
                print(f'rootset `{prefix}`: clean (no imports outside the set)')

    if problems:
        print(f'\nFAILED ({problems} problem(s)); import graph is not a DAG.')
        return 1
    print('OK: import graph is a DAG.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
