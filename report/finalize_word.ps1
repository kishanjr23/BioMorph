# Open the generated report in Microsoft Word, style and update the table of contents,
# save, and export a PDF next to it. Usage: powershell -File report\finalize_word.ps1 <report.docx>
param([Parameter(Mandatory = $true)][string]$Docx)
$path = (Resolve-Path $Docx).Path
$pdf = [System.IO.Path]::ChangeExtension($path, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($path)
    foreach ($id in -20, -21) {                     # wdStyleTOC1, wdStyleTOC2
        $st = $doc.Styles.Item($id)
        $st.Font.Name = "Times New Roman"
        $st.Font.Size = 12
        $st.Font.Bold = [int]($id -eq -20)
        $st.ParagraphFormat.SpaceAfter = 4
    }
    if ($doc.TablesOfContents.Count -ge 1) { $doc.TablesOfContents.Item(1).Update() }
    $doc.Fields.Update() | Out-Null
    if ($doc.TablesOfContents.Count -ge 1) { $doc.TablesOfContents.Item(1).Update() }
    $doc.Save()
    $doc.SaveAs2($pdf, 17)                          # wdFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
}
finally {
    $word.Quit()
}
"saved $path and $pdf"
