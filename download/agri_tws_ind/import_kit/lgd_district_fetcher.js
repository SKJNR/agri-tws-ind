/* ============================================================================
 * LGD DISTRICT-CODE ONE-TIME EXPORT — founder browser fetcher
 * Program: AGRI-TWS-IND | Task: 57-KIT | Built: 2026-09-15 (GLM)
 * ----------------------------------------------------------------------------
 * WHY: lgdirectory.gov.in is captcha-gated for bots (sandbox DWR calls were
 *      rejected — see Q57-A evidence). YOUR browser, after you solve the
 *      captcha once, holds the session the portal trusts. This script uses
 *      the portal's OWN dropdown-filler service (read-only) to pull the
 *      district code list for AP (state code 28) and TG (state code 36).
 *      It fills the `lgd_district_code` column of district_basis.csv (59
 *      rows currently PENDING_FOUNDER_EXPORT, decision D#20).
 *
 * SAFETY: read-only. It never submits the report form, never touches the
 *      captcha field, never writes anything to the portal. Worst case it
 *      fails and prints why.
 *
 * WHERE TO RUN (pick either):
 *   A) https://lgdirectory.gov.in  ->  Reports  ->  "District wise Detail
 *      Report" page (the one with State/District dropdowns + captcha).
 *      Let the page fully load. (Captcha does NOT need to be solved for
 *      this script — the dropdown service is not captcha-gated. If step 5
 *      below says session error, solve the captcha once, then re-paste.)
 *   B) Any other page on lgdirectory.gov.in — the script self-loads the
 *      needed service files. Route A is preferred.
 *
 * HOW: open DevTools (F12) -> Console -> paste this whole file -> Enter.
 *      A JSON file `lgd_district_export_<timestamp>.json` downloads
 *      automatically. Send me that file (or paste its contents in chat).
 *
 * EXPECTED: AP returns 26 districts, TG returns 33 (2026 vintage, D#20
 *      constant target basis). The script warns loudly if counts differ
 *      but still saves whatever it got (fail-loud, never fail-silent).
 * ==========================================================================*/

(async function () {
  'use strict';

  const STATE_CODES = { AP: 28, TG: 36 };
  const EXPECTED = { AP: 26, TG: 33 };        // D#20 basis, 2026 vintage
  const OUT = {
    program: 'AGRI-TWS-IND',
    task: '57-KIT/lgd_district_fetcher',
    generated_at: new Date().toISOString(),
    page_url: location.href,
    portal: 'lgdirectory.gov.in',
    purpose: 'backfill lgd_district_code in d20/district_basis.csv (PENDING_FOUNDER_EXPORT)',
    states: {},
    warnings: [],
    raw_optional: {},
  };

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error('cannot load ' + src));
      document.head.appendChild(s);
    });
  }

  /* --- make sure DWR engine + district service exist (route B support) --- */
  if (typeof dwr === 'undefined' || !dwr.engine) {
    console.log('[lgd-fetch] loading DWR engine...');
    await loadScript('/dwr/engine.js');
  }
  if (typeof lgdDwrDistrictService === 'undefined') {
    console.log('[lgd-fetch] loading lgdDwrDistrictService interface...');
    await loadScript('/dwr/interface/lgdDwrDistrictService.js');
    await sleep(500);
  }
  if (typeof lgdDwrDistrictService === 'undefined') {
    console.error('%c[lgd-fetch] FATAL: lgdDwrDistrictService unavailable. ' +
      'Open the "District wise Detail Report" page (route A), wait for full load, re-paste.',
      'color:red;font-weight:bold');
    return;
  }

  /* --- primary: direct call to the portal's own dropdown service --------- */
  function fetchViaDWR(stateCode) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('DWR timeout (20s)')), 20000);
      try {
        lgdDwrDistrictService.getDistrictList(stateCode, {
          callback: (data) => { clearTimeout(timer); resolve(data); },
          errorHandler: (msg) => { clearTimeout(timer); reject(new Error('DWR error: ' + msg)); },
        });
      } catch (e) { clearTimeout(timer); reject(e); }
    });
  }

  /* --- fallback: drive the page's own dropdown (most human-equivalent) ---- */
  async function fetchViaDropdown(stateCode) {
    const st = document.getElementById('stateList');
    if (!st) throw new Error('no #stateList on this page — use route A page');
    st.value = String(stateCode);
    if (typeof getDistrictList === 'function') { getDistrictList(stateCode); }
    else { st.dispatchEvent(new Event('change')); }
    for (let i = 0; i < 50; i++) {          // poll up to 15 s
      await sleep(300);
      const dl = document.getElementById('districtList');
      if (dl && dl.options && dl.options.length > 1) {
        const out = [];
        for (const o of dl.options) {
          if (o.value && o.value !== '0') out.push({ districtCode: o.value, districtNameEnglish: o.text.trim() });
        }
        if (out.length) return out;
      }
    }
    throw new Error('dropdown never populated (session may need captcha solved once)');
  }

  function normalize(rows) {
    // DWR returns objects like {districtCode, districtNameEnglish, ...}
    const seen = new Set(), out = [];
    for (const r of (rows || [])) {
      const code = String(r.districtCode ?? r.district_code ?? '').trim();
      const name = String(r.districtNameEnglish ?? r.district_name_english ?? r.districtName ?? '').trim();
      if (!code || !name || code === '0' || seen.has(code)) continue;
      seen.add(code);
      out.push({ lgd_district_code: code, district_name_lgd: name });
    }
    out.sort((a, b) => a.lgd_district_code.localeCompare(b.lgd_district_code, undefined, { numeric: true }));
    return out;
  }

  /* ---------------- main loop over AP + TG ---------------- */
  for (const [stName, stCode] of Object.entries(STATE_CODES)) {
    console.log(`%c[lgd-fetch] ${stName} (state code ${stCode}) ...`, 'font-weight:bold');
    let rows = null, route = null, err = null;
    try { rows = await fetchViaDWR(stCode); route = 'dwr-direct'; }
    catch (e1) {
      console.warn('[lgd-fetch] direct DWR failed: ' + e1.message + ' — trying dropdown route');
      try { rows = await fetchViaDropdown(stCode); route = 'dropdown'; }
      catch (e2) { err = 'both routes failed | dwr: ' + e1.message + ' | dropdown: ' + e2.message; }
    }
    if (err) {
      OUT.states[stName] = { state_code: stCode, status: 'FAILED', error: err };
      OUT.warnings.push(`${stName}: ${err}. If it mentions captcha/session: solve the captcha ` +
        'on the page once, then re-paste this script.');
      continue;
    }
    const districts = normalize(rows);
    const n = districts.length;
    const expect = EXPECTED[stName];
    const status = (n === expect) ? 'OK' : `COUNT_MISMATCH (expected ${expect}, got ${n})`;
    OUT.states[stName] = {
      state_code: stCode, status, route, returned_rows: (rows || []).length,
      district_count: n, districts,
    };
    if (n !== expect) {
      OUT.warnings.push(`${stName}: expected ${expect} districts (D#20 2026-vintage basis), got ${n}. ` +
        'Data is saved anyway — a count diff is itself evidence (possible reorg since, or wrong state).');
    }
    console.log(`[lgd-fetch] ${stName}: ${n} districts via ${route} — ${status}`);
    districts.forEach(d => console.log(`   ${String(d.lgd_district_code).padStart(4)}  ${d.district_name_lgd}`));
  }

  /* --- optional richer payloads, kept raw for inspection (never hurt) ---- */
  for (const [label, fn] of [
    ['getDistrictViewList_28', () => lgdDwrDistrictService.getDistrictViewList(28, { callback: d => { OUT.raw_optional[label] = d; }, errorHandler: () => {} })],
    ['getDistrictViewList_36', () => lgdDwrDistrictService.getDistrictViewList(36, { callback: d => { OUT.raw_optional[label] = d; }, errorHandler: () => {} })],
  ]) { try { fn(); } catch (e) { /* optional */ } }
  await sleep(2500);   // give optional callbacks time to land

  /* ---------------- save ---------------- */
  const blob = new Blob([JSON.stringify(OUT, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `lgd_district_export_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a); a.click(); a.remove();

  const ok = Object.entries(OUT.states).filter(([, s]) => s.status === 'OK').length;
  console.log(`%c[lgd-fetch] DONE — ${ok}/2 states OK. JSON downloaded.` +
    (OUT.warnings.length ? ' WARNINGS: ' + OUT.warnings.join(' || ') : ''),
    'color:' + (ok === 2 ? 'green' : 'orange') + ';font-weight:bold');
  console.log('[lgd-fetch] If the download did not start, run: copy(JSON.stringify(OUT))  ' +
    'and paste the clipboard into chat.');
})();
