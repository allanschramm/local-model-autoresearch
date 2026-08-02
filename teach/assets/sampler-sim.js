/**
 * S1D4 restaurant sampler simulator (file:// friendly, no modules).
 * 1 prompt → filtered menu (K/P/Min-P + penalties) → 3 sampled next words.
 */
(function () {
  const BASE_MENU = [
    { w: "telhado", p: 0.28 },
    { w: "muro", p: 0.16 },
    { w: "sofá", p: 0.11 },
    { w: "armário", p: 0.08 },
    { w: "tapete", p: 0.07 },
    { w: "gato", p: 0.06 },
    { w: "chão", p: 0.05 },
    { w: "peito", p: 0.04 },
    { w: "subiu", p: 0.035 },
    { w: "helicóptero", p: 0.03 },
    { w: "planeta", p: 0.025 },
    { w: "abacaxi", p: 0.02 },
  ];

  const els = {
    prompt: document.getElementById("sim-prompt"),
    temp: document.getElementById("temp-range"),
    topk: document.getElementById("topk-range"),
    topp: document.getElementById("topp-range"),
    minp: document.getElementById("minp-range"),
    rep: document.getElementById("rep-range"),
    pres: document.getElementById("pres-range"),
    freq: document.getElementById("freq-range"),
    reshuffle: document.getElementById("reshuffle-btn"),
  };
  if (!els.prompt || !els.temp) return;

  const vals = {
    temp: document.getElementById("temp-val"),
    topk: document.getElementById("topk-val"),
    topp: document.getElementById("topp-val"),
    minp: document.getElementById("minp-val"),
    rep: document.getElementById("rep-val"),
    pres: document.getElementById("pres-val"),
    freq: document.getElementById("freq-val"),
  };
  const menuList = document.getElementById("menu-list");
  const menuCount = document.getElementById("menu-count");
  const orders = [
    document.getElementById("order-1"),
    document.getElementById("order-2"),
    document.getElementById("order-3"),
  ];
  const hint = document.getElementById("sim-hint");
  const configPreview = document.getElementById("config-preview");

  let reshuffleN = 1;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function fold(s) {
    return String(s || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");
  }

  function tokenizePrompt(text) {
    return fold(text).match(/[a-z0-9]+/g) || [];
  }

  function normalize(items) {
    const sum = items.reduce(function (s, it) {
      return s + it.p;
    }, 0);
    if (sum <= 0) {
      return items.map(function (it) {
        return { w: it.w, p: 0 };
      });
    }
    return items.map(function (it) {
      return { w: it.w, p: it.p / sum };
    });
  }

  function applyPenalties(items, promptTokens, rep, pres, freq) {
    const counts = {};
    promptTokens.forEach(function (t) {
      counts[t] = (counts[t] || 0) + 1;
    });
    const recent = promptTokens.slice(-6);
    return items.map(function (it) {
      const key = fold(it.w);
      let score = it.p;
      const n = counts[key] || 0;
      if (n > 0 && pres > 0) score = score * Math.max(0.05, 1 - pres * 0.85);
      if (n > 0 && freq > 0) score = score * Math.max(0.05, 1 - freq * 0.35 * n);
      if (rep > 1 && recent.indexOf(key) !== -1) score = score / rep;
      return { w: it.w, p: score };
    });
  }

  function applyTopK(items, k) {
    if (!k || k <= 0) return items.slice();
    return items.slice(0, k);
  }

  function applyTopP(items, p) {
    if (p >= 0.999) return items.slice();
    let acc = 0;
    const out = [];
    for (let i = 0; i < items.length; i++) {
      out.push(items[i]);
      acc += items[i].p;
      if (acc >= p) break;
    }
    return out.length ? out : items.slice(0, 1);
  }

  function applyMinP(items, minP) {
    if (minP <= 0 || !items.length) return items.slice();
    const top = items[0].p;
    const cut = minP * top;
    const kept = items.filter(function (it) {
      return it.p >= cut;
    });
    return kept.length ? kept : items.slice(0, 1);
  }

  function applyTemp(items, temp) {
    if (!items.length) return items;
    if (temp <= 0.001) {
      const best = items.reduce(function (a, b) {
        return a.p >= b.p ? a : b;
      });
      return items.map(function (it) {
        return { w: it.w, p: it.w === best.w ? 1 : 0 };
      });
    }
    // Softmax(log(p) / temp): temp 1 = cardápio filtrado; >1 achata; <1 afia.
    const logits = items.map(function (it) {
      return Math.log(Math.max(it.p, 1e-12));
    });
    const scaled = logits.map(function (l) {
      return l / temp;
    });
    let maxL = scaled[0];
    for (let i = 1; i < scaled.length; i++) {
      if (scaled[i] > maxL) maxL = scaled[i];
    }
    const exps = scaled.map(function (l) {
      return Math.exp(l - maxL);
    });
    const sum = exps.reduce(function (a, b) {
      return a + b;
    }, 0);
    return items.map(function (it, i) {
      return { w: it.w, p: exps[i] / sum };
    });
  }

  function sampleOne(items, rng) {
    const alive = items.filter(function (it) {
      return it.p > 0;
    });
    if (!alive.length) return items[0] ? items[0].w : "?";
    const r = rng();
    let acc = 0;
    for (let i = 0; i < alive.length; i++) {
      acc += alive[i].p;
      if (r <= acc) return alive[i].w;
    }
    return alive[alive.length - 1].w;
  }

  function hashSeed(parts) {
    let h = 2166136261;
    const s = parts.join("|");
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function runPipeline(settings, promptTokens) {
    let scored = applyPenalties(
      BASE_MENU.map(function (it) {
        return { w: it.w, p: it.p };
      }),
      promptTokens,
      settings.rep,
      settings.pres,
      settings.freq,
    );
    scored = normalize(scored).sort(function (a, b) {
      return b.p - a.p;
    });
    const afterPen = scored.map(function (it) {
      return { w: it.w, p: it.p };
    });

    let kept = applyTopK(scored, settings.topk);
    kept = normalize(kept);
    kept = applyTopP(kept, settings.topp);
    kept = normalize(kept);
    kept = applyMinP(kept, settings.minp);
    kept = normalize(kept);
    const afterFilters = kept.map(function (it) {
      return { w: it.w, p: it.p };
    });
    const forSample = applyTemp(kept, settings.temp);
    return { afterPen: afterPen, afterFilters: afterFilters, forSample: forSample };
  }

  function renderMenu(afterPen, afterFilters) {
    const keptSet = {};
    afterFilters.forEach(function (it) {
      keptSet[it.w] = it.p;
    });
    const maxP = afterFilters.length
      ? Math.max.apply(
          null,
          afterFilters.map(function (it) {
            return it.p;
          }),
        )
      : 1;

    menuList.innerHTML = "";
    afterPen.forEach(function (it) {
      const onTable = Object.prototype.hasOwnProperty.call(keptSet, it.w);
      const showP = onTable ? keptSet[it.w] : 0;
      const row = document.createElement("div");
      row.className = "menu-row" + (onTable ? "" : " cut");
      const name = document.createElement("span");
      name.textContent = it.w;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = onTable ? Math.round((showP / maxP) * 100) + "%" : "0%";
      track.appendChild(fill);
      const pct = document.createElement("span");
      pct.textContent = onTable ? (showP * 100).toFixed(1) + "%" : "fora";
      row.appendChild(name);
      row.appendChild(track);
      row.appendChild(pct);
      menuList.appendChild(row);
    });
    menuCount.textContent = String(afterFilters.length);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function updateSim() {
    const settings = {
      temp: parseFloat(els.temp.value),
      topk: parseInt(els.topk.value, 10),
      topp: parseFloat(els.topp.value),
      minp: parseFloat(els.minp.value),
      rep: parseFloat(els.rep.value),
      pres: parseFloat(els.pres.value),
      freq: parseFloat(els.freq.value),
    };
    const prompt = (els.prompt.value || "").trim() || "O gato subiu no";
    const promptTokens = tokenizePrompt(prompt);

    vals.temp.textContent = settings.temp.toFixed(1);
    vals.topk.textContent = settings.topk === 0 ? "off" : String(settings.topk);
    vals.topp.textContent = settings.topp.toFixed(2);
    vals.minp.textContent = settings.minp.toFixed(2);
    vals.rep.textContent = settings.rep.toFixed(2);
    vals.pres.textContent = settings.pres.toFixed(2);
    vals.freq.textContent = settings.freq.toFixed(2);

    const pipe = runPipeline(settings, promptTokens);
    // Cardápio com % depois da temperatura — senão a ousadia “não muda nada” na lista.
    renderMenu(pipe.afterPen, pipe.forSample);

    const seed = hashSeed([
      prompt,
      settings.temp,
      settings.topk,
      settings.topp,
      settings.minp,
      settings.rep,
      settings.pres,
      settings.freq,
      reshuffleN,
    ]);
    const rng = mulberry32(seed);

    const picks = [];
    for (let i = 0; i < 3; i++) {
      const word = sampleOne(pipe.forSample, rng);
      picks.push(word);
      orders[i].innerHTML =
        escapeHtml(prompt) + ' <span class="picked">' + escapeHtml(word) + "</span>";
    }

    const allSame = picks[0] === picks[1] && picks[1] === picks[2];
    if (settings.temp <= 0.001) {
      hint.textContent = "Temperatura 0: os três pedidos pegam o mesmo prato favorito.";
    } else if (allSame) {
      hint.textContent =
        "Os três coincidiram neste sorteio — aperte “Sortear de novo”. Mesmo ousado, o favorito ainda ganha com mais frequência.";
    } else {
      hint.textContent =
        "Ousadia alta achata as barras do cardápio (pratos fracos sobem), mas o favorito ainda costuma sair mais. “Sortear de novo” mostra outras combinações.";
    }

    if (configPreview) {
      configPreview.textContent =
        "SAMPLER_DEFAULTS = {\n" +
        "    'TEMP': " +
        settings.temp.toFixed(1) +
        ",\n" +
        "    'TOP_P': " +
        settings.topp.toFixed(2) +
        ",\n" +
        "    'TOP_K': " +
        settings.topk +
        ",\n" +
        "    'MIN_P': " +
        settings.minp.toFixed(2) +
        ",\n" +
        "    'REPEAT_PENALTY': " +
        settings.rep.toFixed(2) +
        ",\n" +
        "    'PRESENCE_PENALTY': " +
        settings.pres.toFixed(2) +
        ",\n" +
        "    'FREQUENCY_PENALTY': " +
        (settings.freq === 0 ? "None" : settings.freq.toFixed(2)) +
        ",\n" +
        "}";
    }
  }

  ["temp", "topk", "topp", "minp", "rep", "pres", "freq"].forEach(function (k) {
    els[k].addEventListener("input", updateSim);
  });
  els.prompt.addEventListener("input", updateSim);
  els.reshuffle.addEventListener("click", function () {
    reshuffleN += 1;
    updateSim();
  });
  updateSim();
})();
