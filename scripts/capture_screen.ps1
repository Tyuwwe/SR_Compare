# Capture the full virtual screen (all monitors) to a PNG.
param([Parameter(Mandatory=$true)][string]$OutPath)
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$v = [System.Windows.Forms.SystemInformation]::VirtualScreen
$b = New-Object System.Drawing.Bitmap $v.Width, $v.Height
$g = [System.Drawing.Graphics]::FromImage($b)
$g.CopyFromScreen($v.Left, $v.Top, 0, 0, $b.Size)
$b.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $b.Dispose()
Write-Output "saved $OutPath ($($v.Width)x$($v.Height) from $($v.Left),$($v.Top))"
