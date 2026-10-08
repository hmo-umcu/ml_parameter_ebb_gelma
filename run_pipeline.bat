@echo off
rem ===========================================================================
rem  run_pipeline.bat  -- Windows cmd convenience wrapper for run_pipeline.py
rem
rem  Edit LHS, NCDIR and W once below, then you only ever type the folder path.
rem
rem    run_pipeline.bat data\dev_images\ml_gelma_bioprinting\gelma_deployment\cell_gelma_7_80
rem    run_pipeline.bat data\dev_images\ml_gelma_bioprinting\gelma_deployment --all
rem    run_pipeline.bat data\...\cell_gelma_7_80 --dry_run
rem
rem  Any extra flags you type are passed straight through to run_pipeline.py.
rem  Run it from your project root (the folder that contains "data\").
rem
rem  WHY NCDIR AND NOT NC
rem  --------------------
rem  The sweep range is not the same for every category. Earlier plates were
rem  printed 30-120 kPa (19 wells), later ones 30-140 kPa (23 wells). One
rem  hardcoded NC file fails on half the folders with the misleading message
rem  "23 image(s) on disk but 19 well(s) in the NC".
rem
rem  NCDIR is searched RECURSIVELY, so it can be a broad folder like "data".
rem  Every .nc under it becomes a candidate, and for each sweep folder the one
rem  whose well count matches the number of images is selected. You do not have
rem  to locate the right file or copy it anywhere. Duplicate copies of the same
rem  sweep are fine; identical ones are not treated as ambiguous.
rem
rem  Selection is by PARSED WELL COUNT, never by filename. The file shipped as
rem  pressure_sweap_30120_step5.nc actually contains 23 wells at 30-140 kPa.
rem  Renaming it is worth doing, but nothing here depends on the name.
rem ===========================================================================
 
setlocal
 
rem -- edit these three ------------------------------------------------------
set "LHS=data\lhs_gelma\lhs_bioprint_samples_semicolon.csv"
set "NCDIR=data"
set "W=0.2"
rem --------------------------------------------------------------------------
 
set "SCRIPTS=%~dp0"
 
if "%~1"=="" (
    echo Usage: run_pipeline.bat ^<image_folder^> [extra flags]
    echo        run_pipeline.bat ^<parent_folder^> --all
    echo.
    echo Add --dry_run to see the commands without running them.
    echo Add --sweep_dir ^<name^> if the sweep subfolder is not found.
    exit /b 1
)
 
if not exist "%LHS%" (
    echo [ERROR] LHS CSV not found: %LHS%
    echo         Edit the LHS line at the top of run_pipeline.bat,
    echo         or run this from your project root.
    exit /b 1
)
if not exist "%NCDIR%\" (
    echo [ERROR] Sweep NC folder not found: %NCDIR%
    echo         Edit the NCDIR line at the top of run_pipeline.bat.
    echo         It must be the FOLDER holding your pressure-sweep .nc files,
    echo         not a single file.
    exit /b 1
)
 
python "%SCRIPTS%run_pipeline.py" %* --lhs_csv "%LHS%" --nc_dir "%NCDIR%" --w %W%
exit /b %errorlevel%