param(
    [Parameter(Mandatory = $true)]
    [string]$CKRoot,
    [ValidateSet('all', 'qkv', 'mlp_up', 'mlp_down', 'attn_out')]
    [string]$Shape = 'all',
    [int]$Rows = 9170,
    [int]$Warmup = 3,
    [int]$Iterations = 9,
    [switch]$Check
)

$python = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
$env:PYTHONPATH = $CKRoot
$arguments = @(
    "$PSScriptRoot\convrot_w4a4_bench.py",
    '--ck-root', $CKRoot,
    '--shape', $Shape,
    '--rows', $Rows,
    '--warmup', $Warmup,
    '--iterations', $Iterations
)
if ($Check) {
    $arguments += '--check'
}

& $python @arguments
exit $LASTEXITCODE
