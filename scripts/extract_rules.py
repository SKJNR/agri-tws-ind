import json, re, html

raw = open('/home/z/my-project/competition_main.json').read()
j = json.loads(raw)

def get_text(obj):
    """recursively collect strings"""
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): out.extend(get_text(v))
    elif isinstance(obj, list):
        for v in obj: out.extend(get_text(v))
    return out

texts = get_text(j)
big = max(texts, key=len)  # the HTML page
# strip tags
txt = re.sub(r'<script[\s\S]*?</script>', ' ', big, flags=re.I)
txt = re.sub(r'<style[\s\S]*?</style>', ' ', txt, flags=re.I)
txt = re.sub(r'<[^>]+>', ' ', txt)
txt = html.unescape(txt)
txt = re.sub(r'\s+', ' ', txt)
print("TOTAL TEXT LEN:", len(txt))

# find rules / external data mentions
for kw in ['external', 'External', 'third party', 'Third', 'pre-trained', 'pretrained',
           'Rules', 'rules', 'submission', 'Submissions', 'private', 'Public',
           'trustworth', 'Trustworth', 'innovation', 'Innovation', 'rubric', 'Rubric',
           'GRACE', 'license', 'Licen']:
    for m in re.finditer(re.escape(kw), txt):
        s = max(0, m.start()-350); e = min(len(txt), m.end()+350)
        seg = txt[s:e]
        print(f"\n===== [{kw}] @{m.start()} =====\n{seg}")
        break  # first occurrence only for brevity; full sweep below for key terms

print("\n\n########## ALL 'external data' CONTEXTS ##########")
for m in re.finditer(r'[Ee]xternal', txt):
    s = max(0, m.start()-400); e = min(len(txt), m.end()+400)
    print(f"\n----- @{m.start()} -----\n{txt[s:e]}")
