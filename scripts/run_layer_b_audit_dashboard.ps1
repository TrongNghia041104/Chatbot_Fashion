param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not consistently parse UTF-8 files without a BOM.
# Load the dashboard builder explicitly as UTF-8 so the same command works in
# both Windows PowerShell 5.1 and PowerShell 7+.
$builderPath = Join-Path $PSScriptRoot "build_layer_b_audit_dashboard.ps1"
$builderCode = [System.IO.File]::ReadAllText(
    $builderPath,
    [System.Text.Encoding]::UTF8
)
$builder = [scriptblock]::Create($builderCode)

& $builder -ProjectRoot $ProjectRoot -OutputPath $OutputPath
