@echo off
setlocal

set "DATASET=bindingdb"
rem set "DATASET=celegans"
rem set "DATASET=yourdata"

rem set "radius=0"
rem set "radius=1"
set "radius=2"
rem set "radius=3"

rem set "ngram=2"
set "ngram=3"

python "%~dp0preprocess_data.py" %DATASET% %radius% %ngram%
