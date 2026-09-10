"""Extract human-readable rules text from the scraped competition pages."""
import json, re, html as htmllib

def text_of(path):
    j = json.load(open(path))
    # find the html string anywhere in the structure
    def find_html(o):
        if isinstance(o, str) and '<html' in o[:2000].lower():
            return o
        if isinstance(o, dict):
            for v in o.values():
                r = find_html(v)
                if r: return r
        if isinstance(o, list):
            for v in o:
                r = find_html(v)
                if r: return r
        return None
    h = find_html(j)
    # strip scripts/styles then tags
    h = re.sub(r'<script.*?</script>', ' ', h, flags=re.S | re.I)
    h = re.sub(r'<style.*?</style>', ' ', h, flags=re.S | re.I)
    h = re.sub(r'<[^>]+>', '\n', h)
    h = htmllib.unescape(h)
    h = re.sub(r'[ \t]+', ' ', h)
    h = re.sub(r'\n\s*\n+', '\n', h)
    return h

for path in ['/home/z/my-project/competition_main.json',
             '/home/z/my-project/competition_data.json']:
    t = text_of(path)
    print('=' * 100)
    print('FILE:', path, ' text length:', len(t))
    print('=' * 100)
    # keyword windows
    keys = ['public', 'private', 'split', 'external', 'Copernicus', 'GRACE',
            'GLDAS', 'ERA5', 'dataset', '30', '70', 'leaderboard']
    lines = [l.strip() for l in t.split('\n') if l.strip()]
    seen = set()
    for i, l in enumerate(lines):
        low = l.lower()
        if any(k.lower() in low for k in keys) and len(l) > 30:
            if l not in seen:
                seen.add(l)
                print(f'[{i:5d}] {l[:600]}')
