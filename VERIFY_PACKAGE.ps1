$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrEmpty($PSScriptRoot)) {
    $PSScriptRoot = Get-Location
}
Write-Output "============================================================"
Write-Output "VERIFY PACKAGE - POWERSHELL INTEGRITY AUDIT"
Write-Output "============================================================"
Write-Output "Target package root: $PSScriptRoot"
Write-Output ""

$ready = $true
$missingItems = @()

# A. Required directories
$requiredDirs = @(
    "src",
    "data\processed",
    "data\cache\full",
    "data\cache\kg_complete",
    "data\cache\tvcs_eligible",
    "data\raw\FineFake",
    "data\raw\FineFake\Image",
    "outputs\stage_a_audit",
    "outputs\stage_b_cache_audit",
    "checkpoints\baselines",
    "checkpoints\cikd"
)

Write-Output "Checking required directories..."
foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $PSScriptRoot $dir
    if (Test-Path -Path $fullPath -PathType Container) {
        Write-Output "  [OK] Directory exists: $dir"
    } else {
        Write-Output "  [MISSING] Directory: $dir"
        $ready = $false
        $missingItems += "Directory: $dir"
    }
}
Write-Output ""

# B. Required key files
$requiredFiles = @(
    "data\raw\FineFake\FineFake.pkl",
    "data\processed\manifest_train_seed42.csv",
    "data\processed\manifest_val_seed42.csv",
    "data\processed\manifest_test_seed42.csv",
    "data\processed\manifest_kg_complete_train_seed42.csv",
    "data\processed\manifest_kg_complete_val_seed42.csv",
    "data\processed\manifest_kg_complete_test_seed42.csv",
    "data\processed\manifest_tvcs_eligible_train_seed42.csv",
    "data\processed\manifest_tvcs_eligible_val_seed42.csv",
    "data\processed\manifest_tvcs_eligible_test_seed42.csv",
    "data\cache\relation_vocab.json",
    "outputs\stage_b_cache_audit\01_cache_summary.txt",
    "outputs\stage_b_cache_audit\01_stage_b_smoke_audit_summary.txt"
)

Write-Output "Checking required key files..."
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $PSScriptRoot $file
    if (Test-Path -Path $fullPath -PathType Leaf) {
        Write-Output "  [OK] File exists: $file"
    } else {
        Write-Output "  [MISSING] File: $file"
        $ready = $false
        $missingItems += "File: $file"
    }
}
Write-Output ""

# C. Required cache arrays
$cacheArrays = @{
    "full" = @(
        "text_features.npy",
        "image_features_global.npy",
        "image_features_patch.npy",
        "kg_features.npy",
        "labels_binary.npy",
        "labels_fine.npy",
        "y_ck.npy",
        "split_ids.npy",
        "sample_ids.npy"
    );
    "kg_complete" = @(
        "text_features.npy",
        "image_features_global.npy",
        "image_features_patch.npy",
        "kg_features.npy",
        "relation_ids.npy",
        "labels_binary.npy",
        "labels_fine.npy",
        "y_ck.npy",
        "split_ids.npy",
        "sample_ids.npy"
    );
    "tvcs_eligible" = @(
        "text_features.npy",
        "image_features_global.npy",
        "image_features_patch.npy",
        "kg_features.npy",
        "relation_ids.npy",
        "labels_binary.npy",
        "labels_fine.npy",
        "y_ck.npy",
        "split_ids.npy",
        "sample_ids.npy"
    )
}

Write-Output "Checking required cache arrays..."
foreach ($subset in $cacheArrays.Keys) {
    Write-Output "  Subset: $subset"
    foreach ($arr in $cacheArrays[$subset]) {
        $arrPath = "data\cache\$subset\$arr"
        $fullPath = Join-Path $PSScriptRoot $arrPath
        if (Test-Path -Path $fullPath -PathType Leaf) {
            Write-Output "    [OK] Array: $arr"
        } else {
            # Try to detect equivalent files
            $subsetDir = Join-Path $PSScriptRoot "data\cache\$subset"
            if (Test-Path -Path $subsetDir -PathType Container) {
                $baseWithoutExt = [System.IO.Path]::GetFileNameWithoutExtension($arr)
                $found = Get-ChildItem -Path $subsetDir -Filter "$baseWithoutExt*" | Select-Object -ExpandProperty Name
                if ($found) {
                    Write-Output "    [WARNING] Missing exact '$arr', but found equivalent: $found"
                } else {
                    Write-Output "    [MISSING] Array: $arrPath"
                    $ready = $false
                    $missingItems += "Cache Array: $arrPath"
                }
            } else {
                Write-Output "    [MISSING] Array: $arrPath"
                $ready = $false
                $missingItems += "Cache Array: $arrPath"
            }
        }
    }
}
Write-Output ""

# D. Count FineFake images
Write-Output "Auditing FineFake image count..."
$imageDir = Join-Path $PSScriptRoot "data\raw\FineFake\Image"
if (Test-Path -Path $imageDir -PathType Container) {
    $imageCount = (Get-ChildItem -Path $imageDir -Recurse -File).Count
    Write-Output "  Found $imageCount image files in data\raw\FineFake\Image"
    if ($imageCount -lt 40000) {
        Write-Output "  [WARNING] Image count is $imageCount, which is below the expected ~40,643!"
    } else {
        Write-Output "  [OK] Image count check passed."
    }
} else {
    Write-Output "  [MISSING] Image directory: data\raw\FineFake\Image"
    $imageCount = 0
    $ready = $false
    $missingItems += "Image directory: data\raw\FineFake\Image"
}
Write-Output ""

# E. Compute package size
Write-Output "Computing package folder sizes..."
function Get-FolderSizeGB ($folderPath) {
    if (Test-Path -Path $folderPath -PathType Container) {
        $size = (Get-ChildItem -Path $folderPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
        if ($size -eq $null) { $size = 0 }
        $gb = [Math]::Round($size / 1GB, 3)
        return $gb
    }
    return 0.0
}

$totalSize = Get-FolderSizeGB $PSScriptRoot
Write-Output "  Total package size: $totalSize GB"

$foldersToMeasure = @("data\raw", "data\processed", "data\cache", "data\external", "outputs", "src")
foreach ($folder in $foldersToMeasure) {
    $fullFolder = Join-Path $PSScriptRoot $folder
    $folderSize = Get-FolderSizeGB $fullFolder
    Write-Output "    Size of $folder`: $folderSize GB"
}
Write-Output ""

# F. Verification result
Write-Output "============================================================"
if ($ready) {
    Write-Output "PACKAGE_READY_FOR_STAGE_C_D = TRUE"
} else {
    Write-Output "PACKAGE_READY_FOR_STAGE_C_D = FALSE"
    Write-Output "Missing Items:"
    foreach ($item in $missingItems) {
        Write-Output "  - $item"
    }
}
Write-Output "============================================================"
