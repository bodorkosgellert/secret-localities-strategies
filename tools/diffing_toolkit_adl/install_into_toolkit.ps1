# Install Secret Localities ADL configs into a cloned diffing-toolkit tree.
param(
  [Parameter(Mandatory = $true)]
  [string]$ToolkitPath
)

$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "configs"
$dst = Resolve-Path $ToolkitPath

Copy-Item (Join-Path $src "organism\sl_organism_*.yaml") (Join-Path $dst "configs\organism\") -Force
Copy-Item (Join-Path $src "infrastructure\local_colab.yaml") (Join-Path $dst "configs\infrastructure\") -Force
Copy-Item (Join-Path $src "diffing\method\activation_difference_lens_light.yaml") (Join-Path $dst "configs\diffing\method\") -Force

Write-Host "Installed SL ADL configs into $dst"
Write-Host "See README.md in $PSScriptRoot"
