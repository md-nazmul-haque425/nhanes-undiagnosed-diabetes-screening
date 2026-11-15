# Downloads the 50 NHANES files for the undiagnosed diabetes screening study.
# Files are saved into a sub-folder per cycle next to this script.
$ErrorActionPreference = "Continue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$list = Import-Csv (Join-Path $base "nhanes_file_list.csv")
$ok = 0; $failed = @()
foreach ($row in $list) {
  $dir = Join-Path $base $row.cycle
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $out = Join-Path $dir ($row.file + ".xpt")
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) { Write-Host "Already have" $row.file; $ok++; continue }
  Write-Host "Downloading" $row.cycle $row.file
  try { Invoke-WebRequest -Uri $row.url -OutFile $out -UseBasicParsing; $ok++ }
  catch { Write-Host "  FAILED:" $_.Exception.Message; $failed += $row.url }
}
Write-Host ""
Write-Host "Done:" $ok "of" $list.Count "files."
if ($failed.Count -gt 0) { Write-Host "Failed:"; $failed | ForEach-Object { Write-Host " " $_ } }
Read-Host "Press Enter to close"
