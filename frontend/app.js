/* PathSnap evaluator UI (Laptop 2).
 * Talks only to the PathSnap API: lists trips, sends the selected trip id,
 * and draws exactly what the API returns. No predictions or reference data
 * live in this code.
 * Deep link: index.html?api=http://host:8000#trip=<id> selects and runs a trip. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (v) => (v === null || v === undefined ? "–" : (100 * v).toFixed(1) + "%");
  const int = (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString("en"));
  const plural = (n, word) => `${int(n)} ${word}${n === 1 ? "" : "s"}`;

  const state = { api: "", trips: [], selected: null, busy: false };

  // ---------------------------------------------------------------- map
  const map = L.map("map", { zoomControl: true, preferCanvas: true, attributionControl: true }).setView([20, 0], 2);
  map.zoomControl.setPosition("topright");
  // OpenStreetMap tiles, turned into a dark monochrome basemap by a CSS filter (styles.css).
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  const layers = {
    network: L.layerGroup().addTo(map),
    route: L.layerGroup().addTo(map),
    gps: L.layerGroup().addTo(map),
    markers: L.layerGroup().addTo(map),
  };

  // ---------------------------------------------------------------- api
  async function api(path, opts = {}) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), opts.timeout || 120000);
    try {
      const res = await fetch(state.api + path, {
        ...opts, signal: ctrl.signal,
        headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || "");
        throw new Error(`The server said no (${res.status}). ${detail}`.trim());
      }
      return body;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The server took too long to answer.");
      if (e instanceof TypeError) throw new Error(`Can't reach ${state.api}. Is the API running, and are both laptops on the same network?`);
      throw e;
    } finally { clearTimeout(timer); }
  }

  const showError = (msg) => { $("error").textContent = msg || ""; $("error").hidden = !msg; };
  function setStatus(text, s) { $("api-status").textContent = text; $("api-dot").dataset.state = s; }
  function setBusy(b, text) {
    state.busy = b;
    $("progress").hidden = !b;
    $("loading-text").hidden = !b;
    if (text) $("loading-text").textContent = text;
    if (b) $("empty-note").hidden = true;
    $("run").disabled = b || !state.selected;
  }

  // ---------------------------------------------------------------- connect
  async function connect(url) {
    state.api = url.replace(/\/+$/, "");
    try { localStorage.setItem("pathsnap.api", state.api); } catch (_) { /* storage unavailable */ }
    showError(""); setStatus("Connecting…", "busy");
    try {
      const h = await api("/health", { timeout: 8000 });
      setStatus(h.model_loaded
        ? `Connected. Using ${h.model_name}.`
        : `Connected, but no trained model is loaded, so matching falls back to distance only. ${h.messages.join(" ")}`,
        "ok");
      const t = await api("/trips");
      state.trips = t.trips;
      renderTrips(t.partition);
      const wanted = new URLSearchParams(location.hash.slice(1)).get("trip");
      const btn = wanted && document.querySelector(`.trip[data-id="${CSS.escape(wanted)}"]`);
      if (btn) { await selectTrip(state.trips.find((x) => x.trip_id === wanted), btn); run(); }
    } catch (e) { setStatus("Not connected.", "off"); showError(e.message); }
  }

  function renderTrips(partition) {
    const ol = $("trip-list"); ol.innerHTML = "";
    $("trip-count").textContent = `${state.trips.length} · ${partition.replace("_", " ")}`;
    if (!state.trips.length) { ol.innerHTML = '<li class="empty">The server has no trips to offer.</li>'; return; }
    for (const trip of state.trips) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.className = "trip"; b.type = "button"; b.dataset.id = trip.trip_id;
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = '<span class="id"></span><span class="meta"></span>';
      b.querySelector(".id").textContent = trip.trip_id;
      const mins = Math.max(1, Math.round(trip.duration_s / 60));
      b.querySelector(".meta").textContent = `${int(trip.n_observations)} pts · ${mins} min`;
      b.addEventListener("click", () => selectTrip(trip, b));
      li.appendChild(b); ol.appendChild(li);
    }
  }

  async function selectTrip(trip, button) {
    if (state.busy || !trip) return;
    document.querySelectorAll(".trip").forEach((el) => el.setAttribute("aria-pressed", "false"));
    button.setAttribute("aria-pressed", "true");
    state.selected = trip.trip_id;
    $("results").hidden = true; showError("");
    Object.values(layers).forEach((l) => l.clearLayers());
    const [a, b, c, d] = trip.bbox;
    map.fitBounds([[a, b], [c, d]], { padding: [40, 40] });
    setBusy(true, "Loading the road network…");
    try {
      const net = await api(`/trips/${encodeURIComponent(trip.trip_id)}/network`);
      const road = css("--map-road");
      for (const seg of net.arcs) L.polyline(seg, { color: road, weight: 1, opacity: 0.9, interactive: false }).addTo(layers.network);
    } catch (e) { showError(e.message); }
    finally {
      setBusy(false);
      $("empty-note").hidden = false;
      $("empty-note").textContent = "Ready. Press “Match route”.";
    }
  }

  // ---------------------------------------------------------------- reconstruct
  async function run() {
    if (!state.selected || state.busy) return;
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const compare = $("compare").checked;
    const trip = state.trips.find((t) => t.trip_id === state.selected);
    showError("");
    setBusy(true, `Matching ${int(trip ? trip.n_observations : 0)} points…`);
    try {
      const r = await api("/reconstruct", { method: "POST", body: JSON.stringify({ trip_id: state.selected, mode, compare }) });
      draw(r); renderResults(r);
      $("empty-note").hidden = true;
    } catch (e) { showError(e.message); }
    finally { setBusy(false); }
  }

  function draw(r) {
    layers.route.clearLayers(); layers.gps.clearLayers(); layers.markers.clearLayers();
    const fg = css("--map-ink"), fg2 = css("--map-ink-2"), bg = css("--map-halo");
    for (const line of r.route_polylines) {
      L.polyline(line, { color: bg, weight: 7, opacity: 0.9, interactive: false }).addTo(layers.route); // halo
      L.polyline(line, { color: fg, weight: 3, opacity: 1, lineCap: "round", lineJoin: "round" })
        .bindTooltip("matched route", { sticky: true }).addTo(layers.route);
    }
    const gps = r.observations;
    for (const o of gps) {
      const where = o.arc_id === null ? "no road nearby" : `road ${o.arc_id}`;
      const conf = o.score != null ? ` · p ${o.score.toFixed(2)}` : "";
      L.circleMarker([o.lat, o.lon], { radius: 2.2, stroke: false, fillColor: fg2, fillOpacity: 0.75 })
        .bindTooltip(`#${o.index} · ${where}${conf}`).addTo(layers.gps);
    }
    if (gps.length) {
      const first = gps[0], last = gps[gps.length - 1];
      L.circleMarker([first.lat, first.lon], { radius: 6, color: fg, weight: 2, fillColor: bg, fillOpacity: 1 })
        .bindTooltip("start").addTo(layers.markers);
      L.circleMarker([last.lat, last.lon], { radius: 6, color: bg, weight: 2, fillColor: fg, fillOpacity: 1 })
        .bindTooltip("end").addTo(layers.markers);
      map.fitBounds(L.latLngBounds(gps.map((o) => [o.lat, o.lon])), { padding: [40, 40] });
    }
  }

  function renderResults(r) {
    $("results").hidden = false;
    $("res-trip").textContent = r.trip_id;
    const m = r.metrics;
    $("k-top1").textContent = m ? pct(m.top1_accuracy) : "–";

    const secs = r.timings_ms.request_total_ms / 1000;
    const gaps = r.n_segment_breaks === 0
      ? "The route holds together end to end."
      : `It breaks in ${plural(r.n_segment_breaks, "place")} where the road network didn't connect.`;
    $("summary").textContent =
      `Matched ${int(r.n_matched)} of ${int(r.n_observations_used)} points onto ` +
      `${plural(r.route_arc_ids.length, "road segment")} in ${secs < 1 ? Math.round(secs * 1000) + " ms" : secs.toFixed(1) + " s"}. ${gaps}`;

    const decoder = r.mode === "hmm" ? "classifier + HMM" : "classifier only";
    const facts = [
      ["Decoder", decoder],
      ["Model", r.model_name || "none (distance only)"],
      ["Route continuity", m ? pct(m.route_continuity) : "–"],
      ["Segments right (F1)", m ? pct(m.seq_f1) : "–"],
      ["Length right (prec. / rec.)", m ? `${pct(m.length_precision)} / ${pct(m.length_recall)}` : "–"],
      ["Truth not in candidates", m ? pct(m.candidate_miss_rate) : "–"],
      ["Points used / raw", `${int(r.n_observations_used)} / ${int(r.n_observations_raw)}`],
    ];
    if (!m) facts.push(["Reference data", "not on the server"]);
    const dl = $("details"); dl.innerHTML = "";
    for (const [k, v] of facts) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      dl.append(dt, dd);
    }

    const block = $("compare-block");
    if (!r.comparison) { block.hidden = true; return; }
    block.hidden = false;
    const order = ["classifier", "hmm"].filter((k) => k in r.comparison);
    const name = { classifier: "classifier", hmm: "+ HMM" };
    const rows = [["Top-1", "top1_accuracy"], ["Continuity", "route_continuity"], ["Segments F1", "seq_f1"]];
    const table = $("compare-table"); table.innerHTML = "";
    const head = table.insertRow();
    for (const h of ["", ...order.map((k) => name[k])]) { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); }
    for (const [label, key] of rows) {
      const tr = table.insertRow();
      tr.insertCell().textContent = label;
      const vals = order.map((k) => (r.comparison[k] || {})[key]);
      const best = Math.max(...vals.filter((v) => v != null));
      vals.forEach((v) => {
        const td = tr.insertCell(); td.textContent = pct(v);
        if (v != null && vals.length > 1 && vals.some((x) => x !== v)) td.className = v === best ? "win" : "lose";
      });
    }
  }

  // ---------------------------------------------------------------- init
  let saved = null;
  try { saved = localStorage.getItem("pathsnap.api"); } catch (_) { /* ignore */ }
  const initial = new URLSearchParams(location.search).get("api") || saved || window.PATHSNAP_DEFAULT_API || "http://localhost:8000";
  $("api-url").value = initial;
  $("api-form").addEventListener("submit", (e) => { e.preventDefault(); connect($("api-url").value.trim()); });
  $("run").addEventListener("click", run);
  connect(initial);
})();
