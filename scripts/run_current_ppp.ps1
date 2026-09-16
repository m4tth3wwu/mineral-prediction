$ErrorActionPreference = 'Stop'
$workspace = 'D:\code\ResearchPractice\mineral_prediction'
$env:OMP_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '1'
$runArgs = @($args)
if (-not ($runArgs | Where-Object { $_ -eq '--output-dir' -or $_ -like '--output-dir=*' })) {
    $runArgs += @('--output-dir', (Join-Path $workspace ('output\ppp_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))))
}
Push-Location -LiteralPath $workspace
try {
    & 'D:\anaconda\python.exe' (Join-Path $workspace 'PPP_binary_lithology_v3.py') @runArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
