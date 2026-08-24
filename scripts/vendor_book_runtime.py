#!/usr/bin/env python3
"""Vendor notebook runtime JavaScript and rewrite rendered HTML to local URLs."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, shutil, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
URLS={
    'https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js':'jquery-3.5.1.min.js',
    'https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js':'require-2.3.6.min.js',
    'https://cdn.plot.ly/plotly-3.3.1.min.js':'plotly-3.3.1.min.js',
    'https://cdn.plot.ly/plotly-3.3.1.min':'plotly-3.3.1.min.js',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--book-output',type=Path,default=ROOT/'book/_book');a=p.parse_args()
    out=a.book_output.resolve();vendor=out/'site_libs/vendor';vendor.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.find_spec('plotly')
    if spec is None or spec.origin is None:raise RuntimeError('Plotly is not installed in the active render environment')
    plotly=Path(spec.origin).parent/'package_data/plotly.min.js'
    for url,name in URLS.items():
        dest=vendor/name
        if 'plotly' in name:
            if not plotly.is_file():raise FileNotFoundError(plotly)
            shutil.copyfile(plotly,dest)
        elif not dest.is_file():
            with urllib.request.urlopen(url,timeout=30) as response:dest.write_bytes(response.read())
        if dest.stat().st_size<10_000:raise RuntimeError(f'Vendored runtime unexpectedly small: {dest}')
    replacements={url:f'site_libs/vendor/{name}' for url,name in URLS.items()}
    changed=0
    forbidden=('cdn.jsdelivr.net/npm/jquery','cdn.jsdelivr.net/npm/requirejs','cdn.plot.ly/plotly')
    for html in out.glob('*.html'):
        text=html.read_text()
        revised=text
        for url,local in replacements.items():revised=revised.replace(url,local)
        # Quarto uses local KaTeX for page mathematics. Plotly notebook payloads
        # may still inject legacy remote MathJax loaders; remove those loaders
        # rather than making the deployable site network-dependent.
        revised=re.sub(r'<script[^>]+src="https://cdnjs\.cloudflare\.com/(?:ajax/libs/mathjax|polyfill)/[^" ]+"[^>]*></script>','',revised)
        revised=re.sub(r'<script[^>]+src="https://cdn\.jsdelivr\.net/npm/mathjax[^" ]*"[^>]*></script>','',revised)
        if revised!=text:html.write_text(revised);changed+=1
        if any(x in revised for x in forbidden):raise RuntimeError(f'External notebook runtime remains in {html}')
    manifest={'schema_version':1,'rewritten_html_files':changed,
              'assets':{name:{'bytes':(vendor/name).stat().st_size,'sha256':sha(vendor/name),
                              'source_urls':sorted(u for u,n in URLS.items() if n==name)} for name in sorted(set(URLS.values()))}}
    (vendor/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))
if __name__=='__main__':main()
