Set-Location -Path "C:\Users\Mahra\Desktop\Projects\Thesis\Chapter 6 Advanced MemoryManagementBPTT"
for (($i = 0); $i -lt 5; $i++)
{
    python3 .\experiment.py --memory_mod='No_memory' --trial=$i

}


