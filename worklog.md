# Worklog

---
Task ID: 1
Agent: Super Z (main agent)
Task: Learn/research the Zindi competition "One Step Ahead of Drought: Forecasting Global Water Storage Challenge"

Work Log:
- Fetched leaderboard page: https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/leaderboard
- Fetched main competition info page (overview, rules, evaluation, prizes)
- Fetched competition data page (dataset structure, files)
- Extracted embedded leaderboard table (top 50 + benchmark) from HTML
- Saved raw JSON fetches to /home/z/my-project/leaderboard.json, competition_main.json, competition_data.json

Stage Summary:
- Competition: TWS (Total Water Storage) one-month-ahead global forecasting, organized by ITU/AI for Good with UN partners
- Prize: €2,000 total (1000/600/400); 5,000 Zindi points
- Timeline: 09 Jul 2026 → 13 Sep 2026; 1,070 joined, 414 active
- Data: Train.csv 2,154,021 rows / Test.csv 280,961 rows; features: TWS_t, SPEI (1/3/6/12), SOIL_MOISTURE_t; keyed by (lat, lon, date)
- Target: TWS at t+1; 66.5% of test rows have TWS_t masked → effective horizon 1-7 months
- Metric: RMSE (50%) + Trustworthiness rubric (30%) + Innovation/practicality (20%)
- Rules: 5 subs/day, 200 total, max team 4, no AutoML, open-source only, CC-BY SA 4.0
- Leaderboard top: MOHAR 0.5596, Shankar 0.5893, GIrum 0.6234; benchmark 0.8999

---
Task ID: 2
Agent: Super Z (main agent)
Task: Test whether competition data can be downloaded from Zindi site or Google Drive

Work Log:
- Extracted file download URLs from Zindi data page (api.zindi.world/v1/competitions/.../files/...)
- Tested direct download: HTTP 401 "not authorized" — auth_token=null fails; login required
- Installed gdown 6.1.0 via pip --break-system-packages
- Verified connectivity: drive.usercontent.google.com, docs.google.com, googleapis.com all reachable
- End-to-end test: downloaded gdown's official example file (SPNet model) from Google Drive — successfully pulled 260MB before test timeout stopped it
- Cleaned up test artifacts; confirmed 9.2GB free disk space

Stage Summary:
- Zindi direct download: NOT possible (auth wall, 401)
- Google Drive download: CONFIRMED WORKING (260MB test download succeeded)
- Disk space available: 9.2GB — enough for Train.csv (275MB) + Test.csv (33MB) + workspace
- Recommended workflow: user downloads files from Zindi, uploads to Google Drive with public link, shares link for agent to download
