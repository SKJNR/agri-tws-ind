"""Scan the full Zindi participations LB (all pages) for our scores.
Looking for: 0.677006525 (v27), 0.679780277 (v25), 0.683791578 (v24),
plus general field stats: total rows, freshest best_public_submitted_at,
and the neighborhood around ranks 20-30 to see who is/isn't there.
"""
import json, urllib.request, time

BASE = ("https://api.zindi.world/v1/competitions/"
        "one-step-ahead-of-drought-forecasting-global-water-storage-challenge"
        "/participations")

def fetch(page):
    url = f"{BASE}?page={page}&per_page=50"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

all_rows = []
page = 0
while True:
    d = fetch(page)
    rows = d.get('data', [])
    if not rows:
        break
    all_rows.extend(rows)
    meta = d.get('meta', {})
    page += 1
    if page > 40:
        break
    time.sleep(0.4)

print(f"total participations rows fetched: {len(all_rows)} (pages: {page})")
print(f"meta sample: {json.dumps(meta)[:200]}")

# targets
targets = ['0.677006525', '0.679780277', '0.683791578', '0.687374005',
           '0.693738722', '0.631662555']
found = {}
for r in all_rows:
    u = (r.get('user') or {})
    name = u.get('username', u.get('name', '?')) if isinstance(u, dict) else str(u)
    for t in targets:
        if r.get('best_public_score') == t:
            found[t] = (r.get('public_rank'), name,
                        r.get('submission_count'),
                        r.get('best_public_submitted_at'))
print()
print("TARGET SCORES FOUND:")
for t in targets:
    print(f"  {t}: {found.get(t, 'NOT FOUND in ranked rows')}")

# freshest entries
dated = [(r.get('best_public_submitted_at') or '', r.get('public_rank'),
          r.get('best_public_score')) for r in all_rows]
dated.sort(reverse=True)
print()
print("FRESHEST 8 best-public updates (what the cache includes):")
for ts, rank, sc in dated[:8]:
    print(f"  {ts[:19]}  rank {rank:>4}  {sc}")

# neighborhood ranks 20-30
print()
print("RANKS 20-30 (live API):")
for r in sorted(all_rows, key=lambda x: int(x.get('public_rank') or 10**9)):
    rk = r.get('public_rank')
    if rk and 20 <= int(rk) <= 30:
        u = (r.get('user') or {})
        name = u.get('username', u.get('name', '?')) if isinstance(u, dict) else '?'
        print(f"  rank {rk:>3}  {r.get('best_public_score')}  {name}  "
              f"subs {r.get('submission_count')}  "
              f"{(r.get('best_public_submitted_at') or '')[:16]}")

# save full snapshot
out = [{'rank': r.get('public_rank'), 'score': r.get('best_public_score'),
        'user': (r.get('user') or {}).get('username', '?')
                if isinstance(r.get('user'), dict) else '?',
        'subs': r.get('submission_count'),
        'best_at': r.get('best_public_submitted_at')}
       for r in all_rows]
json.dump(out, open('/home/z/my-project/download/zindi_lb_api_2026-09-08.json', 'w'),
          indent=1)
print()
print("snapshot saved: download/zindi_lb_api_2026-09-08.json")
