#!/bin/bash

echo "OrganicGuard AI - Pest Management System"
echo "====================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    python3 deploy.py setup
    if [ $? -ne 0 ]; then
        echo "Setup failed. Please check the errors above."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if database exists
if [ ! -f "data/pest_analysis.db" ]; then
    echo "Initializing database..."
    python -c "from app import init_database; init_database()"
fi

# Start the application
echo "Starting OrganicGuard AI..."
echo "Open your browser and go to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo
python app.py
