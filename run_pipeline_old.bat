@echo off
rem ===========================================================================
rem  run_pipeline.bat  -- Windows cmd convenience wrapper for run_pipeline.py
rem
rem  Edit LHS, NC and W once below, then you only ever type the folder path.
rem
rem    run_pipeline.bat data\dev_images\ml_gelma_bioprinting\gelma_deployment\cell_gelma_7_80
rem    run_pipeline.bat data\dev_images\ml_gelma_bioprinting\gelma_deployment --all
rem    run_pipeline.bat data\...\cell_gelma_7_80 --dry_run
rem
rem  Any extra flags you type are passed straight through to run_pipeline.py.
rem  Run it from your project root (the folder that contains "data\").
rem ===========================================================================
 
setlocal
 
rem -- edit these three ------------------------------------------------------
set "LHS=data\lhs_gelma\lhs_bioprint_samples_semicolon.csv"
set "NC=data\lhs_gelma\pressure_sweap_30-120_step-5.nc"
set "W=0.2"
rem --------------------------------------------------------------------------
 
set "SCRIPTS=%~dp0"
 
if "%~1"=="" (
    echo Usage: run_pipeline.bat ^<image_folder^> [extra flags]
    echo        run_pipeline.bat ^<parent_folder^> --all
    echo.
    echo Add --dry_run to see the commands without running them.
    exit /b 1
)
 
if not exist "%LHS%" (
    echo [ERROR] LHS CSV not found: %LHS%
    echo         Edit the LHS line at the top of run_pipeline.bat,
    echo         or run this from your project root.
    exit /b 1
)
if not exist "%NC%" (
    echo [ERROR] Sweep NC file not found: %NC%
    echo         Edit the NC line at the top of run_pipeline.bat.
    exit /b 1
)
 
python "%SCRIPTS%run_pipeline.py" %* --lhs_csv "%LHS%" --nc_file "%NC%" --w %W%
exit /b %errorlevel%