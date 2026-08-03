# Guardrail: git commit/push requires human permission.
# Claude Code PreToolUse (Bash|PowerShell).
#
# Policy:
#   git commit / git push (incl. --amend / --force) are DENIED unless a fresh
#   one-shot permission token exists at .claude/hooks/.git-commit-allow
#   (TTL 30 min). Token is human-created only — agents cannot write
#   .claude/hooks/** (permissions.deny + block-gate-tamper.ps1).
#
# Grant (human, PowerShell, repo root):
#   New-Item -ItemType File -Force .claude/hooks/.git-commit-allow | Out-Null
#
# Disable / rollback: docs/discovery/agent-shell-hard-gates.md §3

$ErrorActionPreference = 'Stop'

$ttlMinutes = 30
$tokenName = '.git-commit-allow'
$tokenDir  = $PSScriptRoot   # hook lives in .claude/hooks/ -> token next to it
$tokenPath = Join-Path $tokenDir $tokenName

$denyMsg = @'
BLOCKED by repo hard-gate (.claude/hooks/block-git-commit.ps1).

git commit/push requires your explicit permission. STOP and ask the user first.
Never commit or push without an explicit "commit" / "commit and push" command.

To grant permission for the next 30 minutes (human, PowerShell, repo root):
    New-Item -ItemType File -Force .claude/hooks/.git-commit-allow | Out-Null
Token auto-expires (TTL 30 min). Re-create it for each batch of commits.
Playbook: docs/discovery/agent-shell-hard-gates.md section 3.6
'@

function Emit-Deny([string]$Message, [bool]$IsClaude) {
    if ($IsClaude) {
        [Console]::Error.WriteLine($Message)
        exit 2
    }
    Write-Output (@{ permission = 'deny'; user_message = $Message; agent_message = $Message } | ConvertTo-Json -Compress)
    exit 0
}

function Emit-Allow {
    exit 0
}

$isClaude = $false
$cmd = ''
try {
    [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $raw = [Console]::In.ReadToEnd()
    if ($null -ne $raw) {
        $raw = $raw.Trim([char]0, "`r", "`n", " ", "`t")
    }
    if ([string]::IsNullOrWhiteSpace($raw)) {
        Emit-Allow
    }
    $payload = $raw | ConvertFrom-Json
    if ($null -ne $payload.tool_input) { $isClaude = $true }
    if ($null -ne $payload.tool_input -and $payload.tool_input.command) {
        $cmd = [string]$payload.tool_input.command
    } elseif ($payload.command) {
        $cmd = [string]$payload.command
    }
} catch {
    Emit-Deny "Hook JSON parse failed: $($_.Exception.Message). $denyMsg" $true
}

if ([string]::IsNullOrWhiteSpace($cmd)) { Emit-Allow }

# Match standalone `git commit` / `git push` (word boundary after verb).
# Does NOT match: git log --grep, git commit-tree, git diff, etc.
if ($cmd -notmatch '(?i)\bgit\s+(?:commit|push)(?:\s|$)') {
    Emit-Allow
}

# Git op found -> require a fresh human token.
if (Test-Path -LiteralPath $tokenPath) {
    $last = (Get-Item -LiteralPath $tokenPath).LastWriteTime
    if ((Get-Date) - $last -le [TimeSpan]::FromMinutes($ttlMinutes)) {
        Emit-Allow
    }
}

Emit-Deny $denyMsg $isClaude
