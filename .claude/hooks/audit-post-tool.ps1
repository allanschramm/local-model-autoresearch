# Post-tool audit (Claude Code PostToolUse). Fail-open — never blocks.
# Appends one line per successful tool call for allow/ask tuning.
# Log: .claude/hooks-audit.log (gitignored via *.log).
# Disable / rollback: docs/discovery/agent-shell-hard-gates.md §3

$ErrorActionPreference = 'Continue'

try {
    [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $raw = [Console]::In.ReadToEnd()
    if ($null -ne $raw) {
        $raw = $raw.Trim([char]0, "`r", "`n", " ", "`t")
    }
    if ([string]::IsNullOrWhiteSpace($raw)) { exit 0 }

    $payload = $raw | ConvertFrom-Json
    $tool = ''
    if ($payload.tool_name) { $tool = [string]$payload.tool_name }
    elseif ($payload.tool) { $tool = [string]$payload.tool }

    $detail = ''
    $ti = $payload.tool_input
    if ($null -ne $ti) {
        if ($ti.command) { $detail = [string]$ti.command }
        elseif ($ti.file_path) { $detail = [string]$ti.file_path }
        elseif ($ti.path) { $detail = [string]$ti.path }
    }
    if (-not $detail -and $payload.command) { $detail = [string]$payload.command }

    $detail = ($detail -replace '\s+', ' ').Trim()
    if ($detail.Length -gt 240) { $detail = $detail.Substring(0, 237) + '...' }

    $root = $env:CLAUDE_PROJECT_DIR
    if ([string]::IsNullOrWhiteSpace($root)) {
        $root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    }
    $logDir = Join-Path $root '.claude'
    $logPath = Join-Path $logDir 'hooks-audit.log'
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    $ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $line = "$ts`t$tool`t$detail"
    Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
} catch {
    # Fail open — audit must never break the agent loop.
}

exit 0
