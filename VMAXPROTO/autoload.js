// Drives the VMAX Steward Review prototype from our backend's already-computed
// output files. Nothing here runs a model -- vmax_export.py already ran the
// fast (YOLO) model once, offline, and wrote one prediction JSON per clip;
// this script's only job is: fetch those files, hand them to the prototype's
// own public API (window.VMAX.importPredictions), and not show the review UI
// until that's actually done.
//
// IMPORTANT: the prototype's own addFiles() assigns each clip a random
// crypto.randomUUID() id -- never anything we control. Our backend's clip id
// (a content hash) has no relation to that. The clip's display `name`
// ("Incident N Clip M.mp4") is the only thing both sides agree on and that
// stays stable across reloads, so everything here is correlated by name, not
// by id.
//
// Eager (loads every clip up front) but entirely hidden behind a loading
// gate with a real progress bar, so there's no visible flicker regardless of
// clip count -- the tradeoff from earlier (batch vs. per-clip) no longer
// matters once nothing is on-screen while it happens.
(async () => {
  const gate = document.createElement('div');
  gate.id = 'vmaxLoadingGate';
  gate.style.cssText = 'position:fixed;inset:0;z-index:99999;background:#0a0a0bf5;' +
    'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
    'gap:16px;color:#f5f5f5;font:14px Inter,Arial,sans-serif;';
  gate.innerHTML =
    '<div style="width:34px;height:34px;border:3px solid #2e2e30;border-top-color:#d90d17;' +
    'border-radius:50%;animation:vmaxspin 0.8s linear infinite"></div>' +
    '<div id="vmaxGateText" style="font-size:15px;font-weight:600">Loading clips…</div>' +
    '<div style="width:320px;height:8px;background:#1c1c1d;border:1px solid #2e2e30;border-radius:4px;overflow:hidden">' +
    '<div id="vmaxGateFill" style="height:100%;width:0%;background:#d90d17;transition:width .18s ease-out"></div></div>' +
    '<div id="vmaxGateCount" style="font-size:11px;color:#8c8c91">0 / 0</div>' +
    '<div id="vmaxGateSub" style="font-size:11px;color:#8c8c91;max-width:420px;text-align:center">' +
    'The fast model already ran offline; this is just fetching its saved output.</div>' +
    '<style>@keyframes vmaxspin{to{transform:rotate(360deg)}}</style>';
  document.body.appendChild(gate);
  const setText = (main, sub) => {
    gate.querySelector('#vmaxGateText').textContent = main;
    if (sub !== undefined) gate.querySelector('#vmaxGateSub').textContent = sub;
  };
  const setProgress = (done, total) => {
    gate.querySelector('#vmaxGateFill').style.width = (total ? Math.round(100 * done / total) : 0) + '%';
    gate.querySelector('#vmaxGateCount').textContent = `${done} / ${total}`;
  };

  try {
    const manifest = await (await fetch('/api/clips')).json();
    if (!manifest.clips.length) { setText('No clips available from the backend.', 'Run vmax_export.py, then reload.'); return; }
    const manifestNames = new Set(manifest.clips.map(c => c.name));

    // Self-heal: drop anything the backend no longer lists, and de-dupe by
    // name (name is the only stable cross-session key -- see note above).
    // This is what stops stale/duplicate entries from earlier sessions from
    // accumulating.
    const kept = new Map();
    for (const c of state.clips) if (manifestNames.has(c.name)) kept.set(c.name, c);
    state.clips = Array.from(kept.values());
    const byName = () => new Map(state.clips.map(c => [c.name, c]));

    setProgress(0, manifest.clips.length);
    for (let i = 0; i < manifest.clips.length; i++) {
      const clip = manifest.clips[i];
      setText(`Processing clip ${i + 1} of ${manifest.clips.length}…`, clip.name);

      let c = byName().get(clip.name);
      if (!c) {
        // Never seen this browser before: create it properly via addFiles so
        // it gets a real, correctly-wired clip object (fingerprint, markers,
        // draft, etc.) -- don't construct one by hand.
        const blob = await (await fetch(clip.video_url)).blob();
        await addFiles([new File([blob], clip.name, { type: 'video/mp4' })]);
        c = byName().get(clip.name);
      } else if (!urls.has(c.id)) {
        // Known clip (evidence may already be restored from localStorage),
        // but blob URLs never survive a reload -- reconnect the video only.
        const blob = await (await fetch(clip.video_url)).blob();
        urls.set(c.id, URL.createObjectURL(blob));
      }
      c.fps = clip.fps || c.fps;
      // Every clip with predictions can be sent to the deep model by hand; only
      // low-confidence clips (deep_model_eligible) are sent automatically.
      const useDeep = !clip.playback_only;
      if (!useDeep) delete c.betterModel;
      c.remoteVideoUrl = clip.video_url; c.remotePredictionsUrl = clip.playback_only ? null : clip.predictions_url; c.remoteClipId = useDeep ? clip.id : null;

      if (!clip.playback_only && !analysed(c)) {
        choose(c.id);
        if (!hasVideo()) await new Promise(res => video.addEventListener('loadeddata', res, { once: true }));
        const predictions = await (await fetch(clip.predictions_url)).json();
        window.VMAX.importPredictions(E.validate(predictions));
      } else if (!clip.playback_only) {
        // Predictions saved in this browser can be stale (for example, from
        // before car identity was added). The prototype's importer refuses to
        // replace an original set, so refresh it here from the server -- the
        // source of truth -- and record that in the audit trail.
        const predictions = E.validate(await (await fetch(clip.predictions_url)).json());
        const stored = c.evidence.original;
        if (JSON.stringify([stored.observations, stored.candidates]) !== JSON.stringify([predictions.observations, predictions.candidates])) {
          c.evidence.original = predictions;
          c.evidence.audit.push({ at: new Date().toISOString(), action: 'refresh predictions from server', source: predictions.source, model_version: predictions.model_version });
        }
      }
      // Clips the server auto-escalated (low confidence score) already have a
      // deep-model result waiting; attach it so the badge is right on arrival
      // instead of claiming "Fast model only" for work already done.
      if (useDeep) await refreshBetterModel(c);
      setProgress(i + 1, manifest.clips.length);
    }

    list();
    const first = byName().get(manifest.clips[0].name);
    if (first) { choose(first.id); updatePlaybackStatus(first); }
    gate.remove();
  } catch (err) {
    console.error('VMAX autoload failed', err);
    setText('Could not load clips from the backend.', String((err && err.message) || err));
  }
})();
