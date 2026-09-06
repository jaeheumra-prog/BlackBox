[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$SourceZip,

    [Parameter(Position = 1)]
    [string]$GitPushZip
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$workspaceRoot = Split-Path -Parent $repoRoot

if ([string]::IsNullOrWhiteSpace($SourceZip)) {
    $SourceZip = Join-Path $workspaceRoot 'submit.zip'
}
if ([string]::IsNullOrWhiteSpace($GitPushZip)) {
    $GitPushZip = Join-Path $workspaceRoot 'submit_git_push.zip'
}

$sourceZipPath = (Resolve-Path -LiteralPath $SourceZip).Path
$gitPushZipPath = [System.IO.Path]::GetFullPath($GitPushZip)

if ($sourceZipPath -eq $gitPushZipPath) {
    throw 'SourceZip and GitPushZip must be different files.'
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$managedFiles = @('sync_submit.ps1', 'sync_submit.cmd')
$comparison = [System.StringComparison]::OrdinalIgnoreCase

function Get-SafePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd([char[]]@('\', '/'))
    $normalized = $RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    if ([System.IO.Path]::IsPathRooted($normalized)) {
        throw "Unsafe absolute path in ZIP: $RelativePath"
    }

    $candidate = [System.IO.Path]::GetFullPath((Join-Path $rootPath $normalized))
    $requiredPrefix = $rootPath + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($requiredPrefix, $comparison)) {
        throw "Unsafe path traversal in ZIP: $RelativePath"
    }
    return $candidate
}

function Get-ZipFileNames {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$ValidationRoot
    )

    $names = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $archive.Entries) {
            if ($entry.FullName.EndsWith('/')) {
                continue
            }
            $null = Get-SafePath -Root $ValidationRoot -RelativePath $entry.FullName
            $null = $names.Add($entry.FullName.Replace('\', '/'))
        }
    }
    finally {
        $archive.Dispose()
    }
    # Prevent PowerShell from unrolling the HashSet so Contains() keeps its
    # case-insensitive behavior at the call site.
    return ,$names
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'blackbox-submit-sync-' + [System.Guid]::NewGuid().ToString('N')
)
$tempPackage = $gitPushZipPath + '.' + [System.Guid]::NewGuid().ToString('N') + '.tmp'

New-Item -ItemType Directory -Path $tempRoot | Out-Null

try {
    $sourceNames = Get-ZipFileNames -ZipPath $sourceZipPath -ValidationRoot $tempRoot

    $oldPackageNames = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    if (Test-Path -LiteralPath $gitPushZipPath) {
        $oldPackageNames = Get-ZipFileNames -ZipPath $gitPushZipPath -ValidationRoot $tempRoot
    }

    Expand-Archive -LiteralPath $sourceZipPath -DestinationPath $tempRoot -Force

    $added = 0
    $updated = 0
    $unchanged = 0
    foreach ($entryName in $sourceNames) {
        $sourceFile = Get-SafePath -Root $tempRoot -RelativePath $entryName
        $targetFile = Get-SafePath -Root $repoRoot -RelativePath $entryName

        if (-not (Test-Path -LiteralPath $targetFile -PathType Leaf)) {
            $parent = Split-Path -Parent $targetFile
            if (-not (Test-Path -LiteralPath $parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            Copy-Item -LiteralPath $sourceFile -Destination $targetFile
            $added++
            continue
        }

        $sourceHash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
        $targetHash = (Get-FileHash -LiteralPath $targetFile -Algorithm SHA256).Hash
        if ($sourceHash -ne $targetHash) {
            Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
            $updated++
        }
        else {
            $unchanged++
        }
    }

    $removed = 0
    foreach ($oldEntryName in $oldPackageNames) {
        if ($managedFiles -contains $oldEntryName) {
            continue
        }
        if (-not $sourceNames.Contains($oldEntryName)) {
            $obsoleteFile = Get-SafePath -Root $repoRoot -RelativePath $oldEntryName
            if (Test-Path -LiteralPath $obsoleteFile -PathType Leaf) {
                Remove-Item -LiteralPath $obsoleteFile -Force
                $removed++
            }
        }
    }

    # Rebuild the Git-push package from the untouched source ZIP, then inject
    # these sync scripts. The original submit.zip is only ever opened read-only.
    Copy-Item -LiteralPath $sourceZipPath -Destination $tempPackage
    $package = [System.IO.Compression.ZipFile]::Open(
        $tempPackage,
        [System.IO.Compression.ZipArchiveMode]::Update
    )
    try {
        foreach ($managedFile in $managedFiles) {
            $existingEntries = @($package.Entries | Where-Object {
                $_.FullName -eq $managedFile
            })
            foreach ($existingEntry in $existingEntries) {
                $existingEntry.Delete()
            }

            $managedPath = Join-Path $repoRoot $managedFile
            $null = [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $package,
                $managedPath,
                $managedFile,
                [System.IO.Compression.CompressionLevel]::Optimal
            )
        }
    }
    finally {
        $package.Dispose()
    }

    $verifiedNames = Get-ZipFileNames -ZipPath $tempPackage -ValidationRoot $tempRoot
    foreach ($entryName in $sourceNames) {
        if (-not $verifiedNames.Contains($entryName)) {
            throw "Rebuilt package is missing: $entryName"
        }
    }
    foreach ($managedFile in $managedFiles) {
        if (-not $verifiedNames.Contains($managedFile)) {
            throw "Rebuilt package is missing: $managedFile"
        }
    }

    Move-Item -LiteralPath $tempPackage -Destination $gitPushZipPath -Force

    Write-Host "Sync complete: added=$added updated=$updated removed=$removed unchanged=$unchanged"
    Write-Host "Source ZIP (read-only): $sourceZipPath"
    Write-Host "Git-push ZIP updated: $gitPushZipPath"

    if ((Test-Path -LiteralPath (Join-Path $repoRoot '.git')) -and
        (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host ''
        Write-Host 'Git changes:'
        & git -C $repoRoot status --short
    }
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $tempPackage) {
        Remove-Item -LiteralPath $tempPackage -Force
    }
}
