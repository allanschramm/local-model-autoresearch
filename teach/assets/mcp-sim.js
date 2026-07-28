/**
 * S2D2 — kitchen utensils: built-in tools vs MCP extras (toggle).
 * Classic script for file:// offline. Voice: junior / leigo em IA.
 * Pedido + respostas = metáfora culinária (sem misturar código).
 */
(function () {
  var BUILTIN = [
    { label: "Faca (já no harness)", tool: "cortar" },
    { label: "Panela (já no harness)", tool: "ferver" },
  ];

  var EXTRAS = [
    {
      id: "docs",
      label: "Manual atualizado (Context7)",
      tool: "consultar_manual",
      defaultOn: false,
    },
    {
      id: "calc",
      label: "Balança",
      tool: "pesar",
      defaultOn: false,
    },
  ];

  var QUESTION = "Como faço um chantilly firme que não desanda?";

  function $(id) {
    return document.getElementById(id);
  }

  function stateFromDom() {
    var on = {};
    EXTRAS.forEach(function (p) {
      var el = $("utensil-" + p.id);
      on[p.id] = el ? el.checked : p.defaultOn;
    });
    return on;
  }

  function renderTools(on) {
    var list = $("mcp-tool-list");
    if (!list) return;
    list.innerHTML = "";

    BUILTIN.forEach(function (p) {
      var li = document.createElement("li");
      li.textContent = p.tool + "  ←  " + p.label;
      list.appendChild(li);
    });

    var anyExtra = false;
    EXTRAS.forEach(function (p) {
      if (!on[p.id]) return;
      anyExtra = true;
      var li = document.createElement("li");
      li.textContent = p.tool + "  ←  " + p.label + " (MCP)";
      list.appendChild(li);
    });

    if (!anyExtra) {
      var tip = document.createElement("li");
      tip.className = "muted";
      tip.textContent =
        "Nenhum utensílio extra (MCP) — só o que já vem na cozinha do harness.";
      list.appendChild(tip);
    }
  }

  function renderAnswer(on) {
    var box = $("mcp-answer");
    var badge = $("mcp-answer-badge");
    if (!box || !badge) return;

    // 4 combos — respostas "do cozinheiro", metáfora pura
    if (on.docs && on.calc) {
      badge.textContent = "Manual + balança";
      badge.className = "mcp-badge ok";
      box.innerHTML =
        "<pre class=\"agent-reply\">Pese 200 g de creme de leite bem gelado.\n" +
        "Tigela e batedores frios. Bata em velocidade média\n" +
        "até picos firmes (cerca de 3–5 min). Não aqueça.\n" +
        "Açúcar só no final, se quiser.</pre>";
      return;
    }

    if (on.docs) {
      badge.textContent = "Só manual (Context7)";
      badge.className = "mcp-badge ok";
      box.innerHTML =
        "<pre class=\"agent-reply\">Creme bem gelado. Tigela e batedores frios.\n" +
        "Bata até formar picos firmes. Pare quando\n" +
        "segurar a forma — passar disso vira manteiga.\n" +
        "Nada de micro-ondas nem panela quente.</pre>";
      return;
    }

    if (on.calc) {
      badge.textContent = "Só balança";
      badge.className = "mcp-badge warn";
      box.innerHTML =
        "<pre class=\"agent-reply\">São exatamente 200 g de creme!\n" +
        "Pesei pra você. Agora é só esquentar no fogão\n" +
        "2 minutos que fica firme. Pode confiar.</pre>";
      return;
    }

    badge.textContent = "Nenhum MCP";
    badge.className = "mcp-badge warn";
    box.innerHTML =
      "<pre class=\"agent-reply\">Fácil: joga o creme 3 minutos no micro-ondas\n" +
      "e bate com a faca. Fica rígido na hora.\n" +
      "Todo mundo faz assim.</pre>";
  }

  function refresh() {
    var on = stateFromDom();
    renderTools(on);
    renderAnswer(on);
  }

  function mount() {
    var root = $("mcp-sim");
    if (!root) return;

    var plugs = $("mcp-plugs");
    if (plugs) {
      plugs.innerHTML = "";
      EXTRAS.forEach(function (p) {
        var row = document.createElement("label");
        row.className = "mcp-plug";
        row.htmlFor = "utensil-" + p.id;

        var input = document.createElement("input");
        input.type = "checkbox";
        input.id = "utensil-" + p.id;
        input.checked = p.defaultOn;
        input.addEventListener("change", refresh);

        var text = document.createElement("span");
        text.textContent = p.label;

        var lamp = document.createElement("span");
        lamp.className = "mcp-lamp";
        lamp.setAttribute("aria-hidden", "true");

        row.appendChild(input);
        row.appendChild(lamp);
        row.appendChild(text);
        plugs.appendChild(row);
      });
    }

    var q = $("mcp-question");
    if (q) q.textContent = QUESTION;

    refresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
