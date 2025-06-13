for /f "tokens=2" %%i in ('tasklist ^| findstr %~1') do set pid=%%i
echo PID is: %pid%
ncu --mode attach --set full -f -o NCU_Output %pid%