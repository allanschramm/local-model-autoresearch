/**
 * S2D3 — guardrails sim (file://). Metáfora boate nos textos curtos.
 */
(function () {
  var ACTIONS = [
    {
      id: "read-src",
      label: "Ler src/app.ts",
      matchDeny: false,
      matchAllow: true,
    },
    {
      id: "read-env",
      label: "Ler .env",
      matchDeny: true,
      matchAllow: false,
    },
    {
      id: "bash-ls",
      label: "Bash: ls",
      matchDeny: false,
      matchAllow: true,
    },
    {
      id: "bash-rm",
      label: "Bash: rm -rf /",
      matchDeny: true,
      matchAllow: false,
      hookBlocks: true,
    },
  ];

  function $(id) {
    return document.getElementById(id);
  }

  function flags() {
    return {
      deny: $("gr-deny") && $("gr-deny").checked,
      allow: $("gr-allow") && $("gr-allow").checked,
      pre: $("gr-pre") && $("gr-pre").checked,
      post: $("gr-post") && $("gr-post").checked,
    };
  }

  function decide(a, f) {
    if (f.deny && a.matchDeny) {
      return { cls: "blocked", text: "Deny — bloqueado" };
    }
    if (f.pre && a.hookBlocks) {
      return { cls: "blocked", text: "Pre-hook — barrado na porta" };
    }
    if (f.allow && a.matchAllow) {
      var post = f.post ? " · Post-hook anota depois" : "";
      return { cls: "ok", text: "Allow — passou sem perguntar" + post };
    }
    if (f.post) {
      return {
        cls: "ask",
        text: "Pede confirmação · se rodar, Post-hook anota depois",
      };
    }
    return { cls: "ask", text: "Pede confirmação" };
  }

  function render() {
    var f = flags();
    var list = $("gr-attempt-list");
    var badge = $("gr-mode-badge");
    var summary = $("gr-summary");
    if (!list || !badge || !summary) return;

    var parts = [];
    if (f.deny) parts.push("Deny");
    if (f.allow) parts.push("Allow");
    if (f.pre) parts.push("Pre-hook");
    if (f.post) parts.push("Post-hook");
    badge.textContent = parts.length ? parts.join(" + ") : "Sem guardrail";
    badge.className = "gr-badge " + (parts.length ? "ok" : "warn");

    if (!parts.length) {
      summary.textContent =
        "Sem regra, o harness pergunta de novo. Depois de um tempo, Enter vira reflexo.";
    } else {
      summary.textContent =
        "Deny/Allow = lista. Pre-hook ainda pode barrar. Post-hook só registra.";
    }

    list.innerHTML = "";
    ACTIONS.forEach(function (a) {
      var d = decide(a, f);
      var li = document.createElement("li");
      li.className = "gr-row " + d.cls;
      li.innerHTML =
        "<span class=\"gr-cmd\">" +
        a.label +
        "</span><span class=\"gr-res\">" +
        d.text +
        "</span>";
      list.appendChild(li);
    });
  }

  function bind() {
    ["gr-deny", "gr-allow", "gr-pre", "gr-post"].forEach(function (id) {
      var el = $(id);
      if (el) el.addEventListener("change", render);
    });
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
