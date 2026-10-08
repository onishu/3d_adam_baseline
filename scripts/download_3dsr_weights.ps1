# Download 3DSR pretrained VQ (DADA) weights into third_party/3DSR/checkpoints.
# Does NOT start training. Run manually when ready for Phase 5 GPU work.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DestDir = Join-Path $RepoRoot "third_party\3DSR\checkpoints"
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

$Base = "https://raw.githubusercontent.com/VitjanZ/3DSR/6f83317837c6615f893f3107235c75edda90181a/checkpoints"
$Files = @(
    "DADA_D.pckl",
    "DADA_RGB_D.pckl"
)

foreach ($name in $Files) {
    $url = "$Base/$name"
    $out = Join-Path $DestDir $name
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1MB)) {
        Write-Host "Skip existing $out"
        continue
    }
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $out
}

Write-Host "SHA-256:"
Get-ChildItem $DestDir -Filter "*.pckl" | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
    "{0}  {1}" -f $hash, $_.Name
}
