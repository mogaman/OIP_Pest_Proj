@echo off
echo.
echo 🌱 Organic Pest Classification - Training Scripts
echo ================================================
echo.
echo Select a training script to run:
echo.
echo 1. EfficientNetV2M (Recommended - Best balance)
echo 2. Simple CNN (Fast training for testing)
echo 3. ConvNeXt (Highest accuracy)
echo 4. View README
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting EfficientNetV2M training...
    python train_efficientnet.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🏃 Starting Simple CNN training...
    python train_model_simple.py
    pause
) else if "%choice%"=="3" (
    echo.
    echo 🎯 Starting ConvNeXt training...
    python train_convnext.py
    pause
) else if "%choice%"=="4" (
    echo.
    echo 📖 Opening README...
    start README.md
) else if "%choice%"=="5" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto start
)

:start
goto :eof
