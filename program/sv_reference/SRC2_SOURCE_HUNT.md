# 57-GWL-SRC2 — seven-way source hunt for gwl_data.csv (2026-09-15)

All avenues verified live, all dead from this sandbox:
1. GitHub repo data/ — authors exclude training data by design (verbatim
   in their data/README.md).
2. git ls-remote — only refs/heads/main exists; no data branches.
3. releases.atom — zero releases, no assets.
4. No .gitattributes — no Git LFS objects.
5. HuggingFace datasets API (search: soulvision, ground_water_level) —
   no mirror.
6. AIKosh live via headless browser — page HTML loads (200) but CDN
   returns the HTML shell for every /web/*.js chunk (verified via
   in-page fetch()); console: "unsupported MIME type ('text/html')";
   Angular app cannot boot from overseas IPs -> no download URL
   discoverable programmatically.
7. Wayback Machine — unreachable from sandbox.

Conclusion: one-time founder download from AIKosh (works from India) is
the only acquisition path. ARRIVAL=COMMIT chain pre-built and waiting.
