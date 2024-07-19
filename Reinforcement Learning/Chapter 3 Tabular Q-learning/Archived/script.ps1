$a = 'smallCorridor'
$b = 'crossmap'
$c = 'Tshaped'
$d = 'LongCorridor'
$e = @($a,$b,$c,$d)
$txt = ".txt"
ForEach($j in $e )
{
	$out = "$j$txt"
	For ($i = 0; $i -le 5; $i++)
	{

		python3 DQRN.py --tabularQ --mapname=$j 2> err.txt >>$out
		python3 DQRN.py --mapname=$j 2> err.txt >>$out
		python3 DQRN.py --tabularQ --addmemory --mapname=$j 2> err.txt >>$out
		python3 DQRN.py --addmemory --mapname=$j 2> err.txt >>$out
	}
}