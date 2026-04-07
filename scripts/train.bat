@echo off
chcp 936 >nul 2>&1
setlocal enabledelayedexpansion

:: ============================================================================
:: FaceSR 训练脚本
::
:: 命令：
::   train.bat build
::   train.bat start [额外参数]
::   train.bat resume [额外参数]
::   train.bat status
::   train.bat clean
::   train.bat help
:: ============================================================================

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"
set "BIN_DIR=%BUILD_DIR%\bin\Release"
set "EXE=%BIN_DIR%\facesr_train.exe"
set "CONFIG=%PROJECT_DIR%config\train_config.ini"
set "CHECKPOINT_DIR=%PROJECT_DIR%checkpoints"

if "%~1"=="" goto :menu
if /i "%~1"=="build"  goto :build
if /i "%~1"=="start"  goto :start
if /i "%~1"=="resume" goto :resume
if /i "%~1"=="status" goto :status
if /i "%~1"=="clean"  goto :clean
if /i "%~1"=="help"   goto :help
echo [错误] 未知命令：%~1
goto :help

:build
echo.
echo ========================================
echo   构建 FaceSR 训练程序
echo ========================================
echo.

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [1/2] 正在执行 CMake 配置...
cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF
if errorlevel 1 (
    echo.
    echo [错误] CMake 配置失败。
    pause
    exit /b 1
)

echo.
echo [2/2] 正在编译 facesr_train...
cmake --build "%BUILD_DIR%" --config Release --target facesr_train --parallel
if errorlevel 1 (
    echo.
    echo [错误] 编译失败。
    pause
    exit /b 1
)

echo.
echo [完成] 构建成功：%EXE%
pause
exit /b 0

:start
echo.
echo ========================================
echo   启动 FaceSR 训练
echo ========================================
echo.

if not exist "%EXE%" (
    echo [信息] 未找到 facesr_train.exe，先执行构建...
    call :build
    if errorlevel 1 exit /b 1
    echo.
)

set "AUTO_RESUME_ARGS="
if exist "%CHECKPOINT_DIR%\generator_latest.pt" (
    set "AUTO_RESUME_ARGS=--resume ""%CHECKPOINT_DIR%"""
    echo [信息] 检测到历史检查点，start 将自动从最新状态继续训练。
    if exist "%CHECKPOINT_DIR%\train_state.bin" (
        echo [信息] 检测到训练状态文件：%CHECKPOINT_DIR%\train_state.bin
    )
    echo.
)

if not exist "%CONFIG%" (
    echo [警告] 未找到配置文件：%CONFIG%
    echo [警告] 将仅使用程序默认参数和命令行参数运行。
    echo.
)

set "EXTRA_ARGS="
set "SKIP=1"
for %%a in (%*) do (
    if !SKIP! equ 0 (
        set "EXTRA_ARGS=!EXTRA_ARGS! %%a"
    )
    set "SKIP=0"
)

echo [配置] %CONFIG%
echo [检查点目录] %CHECKPOINT_DIR%
echo [提示] 训练中按 Ctrl+C 可安全保存并停止。
echo.

if exist "%CONFIG%" (
    "%EXE%" --config "%CONFIG%" %AUTO_RESUME_ARGS% %EXTRA_ARGS%
) else (
    "%EXE%" %AUTO_RESUME_ARGS% %EXTRA_ARGS%
)

set "EXIT_CODE=%errorlevel%"
echo.
if %EXIT_CODE% equ 0 (
    echo [完成] 训练结束。
) else (
    echo [失败] 训练退出码：%EXIT_CODE%
)
pause
exit /b %EXIT_CODE%

:resume
echo.
echo ========================================
echo   恢复 FaceSR 训练
echo ========================================
echo.

if not exist "%EXE%" (
    echo [信息] 未找到 facesr_train.exe，先执行构建...
    call :build
    if errorlevel 1 exit /b 1
    echo.
)

if not exist "%CHECKPOINT_DIR%" (
    echo [错误] 检查点目录不存在：%CHECKPOINT_DIR%
    pause
    exit /b 1
)

set "MISSING=0"
if not exist "%CHECKPOINT_DIR%\generator_latest.pt" (
    echo [错误] 缺少文件：%CHECKPOINT_DIR%\generator_latest.pt
    set "MISSING=1"
)
if not exist "%CHECKPOINT_DIR%\discriminator_latest.pt" (
    echo [警告] 缺少文件：%CHECKPOINT_DIR%\discriminator_latest.pt
)
if not exist "%CHECKPOINT_DIR%\train_state.bin" (
    echo [警告] 缺少文件：%CHECKPOINT_DIR%\train_state.bin（可能会从 epoch 0 开始）
)
if %MISSING% equ 1 (
    pause
    exit /b 1
)

call :show_status_quiet

set "EXTRA_ARGS="
set "SKIP=1"
for %%a in (%*) do (
    if !SKIP! equ 0 (
        set "EXTRA_ARGS=!EXTRA_ARGS! %%a"
    )
    set "SKIP=0"
)

echo.
echo [提示] 训练中按 Ctrl+C 可安全保存并停止。
echo.

if exist "%CONFIG%" (
    "%EXE%" --config "%CONFIG%" --resume "%CHECKPOINT_DIR%" %EXTRA_ARGS%
) else (
    "%EXE%" --resume "%CHECKPOINT_DIR%" %EXTRA_ARGS%
)

set "EXIT_CODE=%errorlevel%"
echo.
if %EXIT_CODE% equ 0 (
    echo [完成] 训练结束。
) else (
    echo [失败] 训练退出码：%EXIT_CODE%
)
pause
exit /b %EXIT_CODE%

:status
echo.
echo ========================================
echo   FaceSR 检查点状态
echo ========================================
echo.

if not exist "%CHECKPOINT_DIR%" (
    echo [信息] 尚未发现检查点目录：%CHECKPOINT_DIR%
    pause
    exit /b 0
)

call :show_status_detail
pause
exit /b 0

:show_status_quiet
echo.
echo --- 检查点文件 ---
if exist "%CHECKPOINT_DIR%\train_state.bin"         echo [有] train_state.bin
if exist "%CHECKPOINT_DIR%\generator_latest.pt"     echo [有] generator_latest.pt
if exist "%CHECKPOINT_DIR%\discriminator_latest.pt" echo [有] discriminator_latest.pt
if exist "%CHECKPOINT_DIR%\optimizer_g_state.pt"    echo [有] optimizer_g_state.pt
if exist "%CHECKPOINT_DIR%\optimizer_d_state.pt"    echo [有] optimizer_d_state.pt
goto :eof

:show_status_detail
echo 路径：%CHECKPOINT_DIR%
echo.
echo --- 文件列表 ---
for %%f in ("%CHECKPOINT_DIR%\*.pt" "%CHECKPOINT_DIR%\*.bin") do (
    if exist "%%~f" echo   %%~nxf
)
echo.
echo --- 恢复训练必需文件 ---
if exist "%CHECKPOINT_DIR%\train_state.bin"         (echo [OK] train_state.bin) else (echo [缺少] train_state.bin)
if exist "%CHECKPOINT_DIR%\generator_latest.pt"     (echo [OK] generator_latest.pt) else (echo [缺少] generator_latest.pt)
if exist "%CHECKPOINT_DIR%\discriminator_latest.pt" (echo [OK] discriminator_latest.pt) else (echo [缺少] discriminator_latest.pt)
if exist "%CHECKPOINT_DIR%\optimizer_g_state.pt"    (echo [OK] optimizer_g_state.pt) else (echo [缺少] optimizer_g_state.pt)
if exist "%CHECKPOINT_DIR%\optimizer_d_state.pt"    (echo [OK] optimizer_d_state.pt) else (echo [缺少] optimizer_d_state.pt)
goto :eof

:clean
echo.
if exist "%BUILD_DIR%" (
    set /p "CONFIRM=确认删除构建目录 %BUILD_DIR% ? (y/N): "
    if /i "!CONFIRM!"=="y" (
        rmdir /s /q "%BUILD_DIR%"
        echo [完成] 构建目录已删除。
    ) else (
        echo 已取消。
    )
) else (
    echo [信息] 构建目录不存在。
)
pause
exit /b 0

:help
echo.
echo ============================================
echo   FaceSR 训练脚本
echo ============================================
echo.
echo 用法：
echo   train.bat ^<命令^> [额外参数]
echo.
echo 命令：
echo   build     构建 facesr_train
echo   start     启动训练（如有历史检查点会自动续训）
echo   resume    从 checkpoints 目录恢复训练
echo   status    查看检查点状态
echo   clean     删除构建目录
echo   help      显示帮助
echo.
echo 示例：
echo   train.bat build
echo   train.bat start
echo   train.bat start --epochs 100 --batch-size 8
echo   train.bat resume
echo   train.bat resume --epochs 500
echo   train.bat status
echo.
pause
exit /b 0

:menu
echo.
echo ============================================
echo   FaceSR 训练脚本
echo ============================================
echo.
echo   [1] build
echo   [2] start
echo   [3] resume
echo   [4] status
echo   [5] clean
echo   [0] exit
echo.
set /p "CHOICE=请选择 (0-5): "

if "%CHOICE%"=="1" goto :build
if "%CHOICE%"=="2" goto :start
if "%CHOICE%"=="3" goto :resume
if "%CHOICE%"=="4" goto :status
if "%CHOICE%"=="5" goto :clean
if "%CHOICE%"=="0" exit /b 0

echo [错误] 无效选择：%CHOICE%
goto :menu
