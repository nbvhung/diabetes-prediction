@echo off
chcp 65001 >nul
title Diabetes Prediction - Expo Dev Server
cd /d "%~dp0"
echo ============================================
echo  APP TIEU DUONG - dang khoi dong server...
echo  Quet QR bang app Expo Go tren dien thoai
echo  Nhan Ctrl+C de tat khi khong dung nua
echo ============================================
call npx expo start --tunnel
pause
