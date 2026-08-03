/**
 * Guardrail: git commit/push requires human permission (pi agent).
 *
 * Blocks `git commit` / `git push` in the bash tool unless the user confirms
 * in the TUI. Headless (no UI) => always block. Mirrors
 * .claude/hooks/block-git-commit.ps1 for Claude Code sessions.
 *
 * Disable: delete this file (or remove .pi/extensions).
 * Docs: docs/discovery/agent-shell-hard-gates.md section 3.7
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const GIT_MUTATING = /\bgit\s+(?:commit|push)(?:\s|$)/i;

export default function (pi: ExtensionAPI) {
	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "bash") return undefined;

		const command = event.input.command as string;
		if (!GIT_MUTATING.test(command)) return undefined;

		if (!ctx.hasUI) {
			return {
				block: true,
				reason:
					"git commit/push blocked: no UI to confirm. Ask the user for explicit permission before any commit/push.",
			};
		}

		const allowed = await ctx.ui.confirm(
			"git commit/push?",
			`Allow this command?\n\n  ${command}\n\n(AGENTS.md: never commit without explicit user command)`,
		);
		if (!allowed) return { block: true, reason: "git commit/push denied by user" };

		return undefined;
	});
}
