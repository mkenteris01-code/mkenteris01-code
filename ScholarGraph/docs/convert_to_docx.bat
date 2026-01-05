@echo off
REM Script to convert MDPI manuscript to DOCX format
REM Uses portable pandoc from ../tools/pandoc-3.5/

set INPUT=%~dp0FL-KG-LLM_Scoping_Review_MDPI_Manuscript.md
set OUTPUT=%~dp0FL-KG-LLM_Scoping_Review_MDPI_Manuscript.docx
set PANDOC=%~dp0..\tools\pandoc-3.5\pandoc.exe

echo Converting %INPUT% to %OUTPUT%
echo.

REM Check if portable pandoc exists
if not exist "%PANDOC%" (
    echo ERROR: pandoc not found at %PANDOC%
    echo.
    echo Please download pandoc portable from: https://github.com/jgm/pandoc/releases
    echo and extract to: %~dp0..\tools\pandoc-3.5\
    echo.
    echo Alternatively, install system-wide pandoc from: https://pandoc.org/installing.html
    pause
    exit /b 1
)

REM Convert to DOCX with MDPI-style formatting (if template exists)
if exist "%~dp0MDPI_template.docx" (
    echo Using MDPI template...
    "%PANDOC%" "%INPUT%" -o "%OUTPUT%" --reference-doc="%~dp0MDPI_template.docx"
) else (
    echo Using default DOCX formatting...
    echo (Note: MDPI_template.docx not found - using default style)
    "%PANDOC%" "%INPUT%" -o "%OUTPUT%" -f markdown -t docx
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! File saved to:
    echo %OUTPUT%
) else (
    echo.
    echo Conversion failed. Check file paths.
)

echo.
pause
