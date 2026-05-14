# Shortcut functions for the analyzeToVec tools/. Dot-source this file
# from a PowerShell session to register the shortcuts:
#
#   PS> . C:\Users\102460\Desktop\senao\analyzeToVec\tools\aliases.ps1
#
# To make the shortcuts permanent, add the same dot-source line to your
# PowerShell profile (see `$PROFILE.CurrentUserAllHosts`). The line
# below is repo-rooted via $PSScriptRoot, so the shortcuts keep working
# even if you cd elsewhere.

$_AV_ROOT = Split-Path -Parent $PSScriptRoot

function vectorize    { py (Join-Path $_AV_ROOT 'vectorize.py')        @args }
function preview      { py (Join-Path $_AV_ROOT 'preview.py')          @args }
function regression   { py (Join-Path $_AV_ROOT 'tools\regression.py') @args }
function ablation     { py (Join-Path $_AV_ROOT 'tools\ablation.py')   @args }
function audit-view   { py (Join-Path $_AV_ROOT 'tools\audit_view.py') @args }

Write-Host "analyzeToVec shortcuts registered:" -ForegroundColor Green
Write-Host "  vectorize  preview  regression  ablation  audit-view"
Write-Host "  (root: $_AV_ROOT)"
