@echo off
echo OrganicGuard AI - Pest Management System
echo =====================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Running setup...
    python deploy.py setup
    if errorlevel 1 (
        echo Setup failed. Please check the errors above.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if database exists
if not exist "data\pest_analysis.db" (
    echo Initializing database...
    python -c "from app import init_database; init_database()"
)

REM Start the application
echo Starting OrganicGuard AI...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
