# 超分辨率人脸图像重建 - C++实现
# CMake构建配置 (优化版)

cmake_minimum_required(VERSION 3.18)
project(FaceSR VERSION 1.0.0 LANGUAGES CXX)

# ==================== 编译选项 ====================

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA支持
# 项目不编译.cu文件, 仅链接预编译的LibTorch CUDA库
# 设置TORCH_CUDA_SKIP_NVCC跳过nvcc编译器检测 (解决CUDA 11.8 + MSVC 19.44兼容性问题)
option(TORCH_CUDA_SKIP_NVCC "Skip CUDA compiler detection (for pre-built LibTorch)" ON)
if(TORCH_CUDA_SKIP_NVCC)
    message(STATUS "CUDA nvcc compiler detection: SKIPPED (using pre-built LibTorch CUDA)")
    set(CUDA_AVAILABLE TRUE)
endif()

# 编译优化选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MSVC UTF-8 源文件编码支持
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Release模式优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    endif()
endif()

# Debug模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    endif()
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 导出compile_commands.json用于IDE支持
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==================== 查找依赖库 ====================

# LibTorch (PyTorch C++ API)
if(NOT DEFINED CMAKE_PREFIX_PATH)
    if(WIN32)
        # Windows默认路径
        if(EXISTS "$ENV{USERPROFILE}/libtorch")
            set(CMAKE_PREFIX_PATH "$ENV{USERPROFILE}/libtorch")
        elseif(EXISTS "C:/libtorch")
            set(CMAKE_PREFIX_PATH "C:/libtorch")
        endif()
    else()
        # Linux/Mac默认路径
        if(EXISTS "$ENV{HOME}/libtorch")
            set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
        elseif(EXISTS "/usr/local/libtorch")
            set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
        endif()
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 设置 protobuf 为已找到，避免 OpenCV 重复查找
set(Protobuf_FOUND TRUE)
set(PROTOBUF_FOUND TRUE)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# Qt (可选，仅GUI需要) - 支持 Qt5 和 Qt6
option(BUILD_GUI "Build GUI application" ON)
if(BUILD_GUI)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    # 优先尝试 Qt6，如果失败则尝试 Qt5
    find_package(Qt6 COMPONENTS Core Widgets Gui QUIET)
    if(Qt6_FOUND)
        message(STATUS "Found Qt6, using Qt6")
        set(QT_VERSION 6)
        set(QT_LIBRARIES Qt6::Core Qt6::Widgets Qt6::Gui)
    else()
        find_package(Qt5 COMPONENTS Core Widgets Gui QUIET)
        if(Qt5_FOUND)
            message(STATUS "Found Qt5, using Qt5")
            set(QT_VERSION 5)
            set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::Gui)
        else()
            message(WARNING "Qt5/Qt6 not found, GUI will not be built")
            set(BUILD_GUI OFF)
        endif()
    endif()
endif()

# ==================== 头文件目录 ====================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# ==================== 源文件 ====================

# 通用头文件
set(COMMON_HEADERS
    include/common/config.h
    include/common/logger.h
    include/common/random.h
    include/common/config_parser.h
)

# 模型源文件
set(MODEL_SOURCES
    src/models/generator.cpp
    src/models/discriminator.cpp
    src/models/losses.cpp
)

# 工具源文件
set(UTILS_SOURCES
    src/utils/dataset.cpp
    src/utils/image_utils.cpp
    src/utils/metrics.cpp
)

# GUI源文件
if(BUILD_GUI)
    set(GUI_SOURCES
        src/gui/main_window.cpp
        src/gui/style_config.cpp
        src/gui/background_widget.cpp
        src/gui/frosted_widget.cpp
    )

    # 显式添加头文件以确保 MOC 处理
    set(GUI_HEADERS
        include/gui/main_window.h
        include/gui/style_config.h
        include/gui/background_widget.h
        include/gui/frosted_widget.h
    )
endif()

# ==================== 构建目标 ====================

# 模型库
add_library(facesr_models STATIC ${MODEL_SOURCES})
target_link_libraries(facesr_models PUBLIC ${TORCH_LIBRARIES})
target_include_directories(facesr_models PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 工具库
add_library(facesr_utils STATIC ${UTILS_SOURCES})
target_link_libraries(facesr_utils PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(facesr_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

# GUI库 (可选)
if(BUILD_GUI)
    add_library(facesr_gui STATIC ${GUI_SOURCES} ${GUI_HEADERS})
    target_link_libraries(facesr_gui PUBLIC
        ${QT_LIBRARIES}
        facesr_models facesr_utils
    )
    target_include_directories(facesr_gui PUBLIC ${CMAKE_SOURCE_DIR}/include)

    # 确保 MOC 能找到头文件
    set_target_properties(facesr_gui PROPERTIES
        AUTOMOC ON
        AUTOMOC_MOC_OPTIONS "-I${CMAKE_SOURCE_DIR}/include"
    )
endif()

# 训练程序
add_executable(facesr_train
    src/main_train.cpp
    src/trainer.cpp
)
target_link_libraries(facesr_train PRIVATE
    facesr_models facesr_utils
    ${TORCH_LIBRARIES} ${OpenCV_LIBS}
)

# 测试/推理程序
add_executable(facesr_test
    src/main_test.cpp
    src/inference.cpp
)
target_link_libraries(facesr_test PRIVATE
    facesr_models facesr_utils
    ${TORCH_LIBRARIES} ${OpenCV_LIBS}
)

# GUI程序 (可选)
if(BUILD_GUI)
    add_executable(facesr_gui_app
        src/main_gui.cpp
        src/inference.cpp
    )
    target_link_libraries(facesr_gui_app PRIVATE
        facesr_gui facesr_models facesr_utils
        ${TORCH_LIBRARIES} ${OpenCV_LIBS}
        ${QT_LIBRARIES}
    )
    # 在Windows上隐藏控制台窗口
    if(WIN32)
        set_target_properties(facesr_gui_app PROPERTIES WIN32_EXECUTABLE TRUE)
    endif()
endif()

# ==================== 编译器警告设置 ====================

if(MSVC)
    target_compile_options(facesr_models PRIVATE /W4 /WX-)
    target_compile_options(facesr_utils PRIVATE /W4 /WX-)
    target_compile_options(facesr_train PRIVATE /W4 /WX-)
    target_compile_options(facesr_test PRIVATE /W4 /WX-)
    if(BUILD_GUI)
        target_compile_options(facesr_gui PRIVATE /W4 /WX-)
        target_compile_options(facesr_gui_app PRIVATE /W4 /WX-)
    endif()
else()
    target_compile_options(facesr_models PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(facesr_utils PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(facesr_train PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(facesr_test PRIVATE -Wall -Wextra -Wpedantic)
    if(BUILD_GUI)
        target_compile_options(facesr_gui PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(facesr_gui_app PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ==================== Windows特定配置 ====================

if(WIN32)
    # 复制LibTorch DLL到输出目录 (排除Qt5 conda DLL, 避免与Qt6冲突)
    file(GLOB TORCH_DLLS_ALL "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    set(TORCH_DLLS "")
    foreach(dll ${TORCH_DLLS_ALL})
        get_filename_component(dll_name "${dll}" NAME)
        if(NOT dll_name MATCHES "^Qt5.*")
            list(APPEND TORCH_DLLS "${dll}")
        endif()
    endforeach()

    # 为训练程序复制DLL
    add_custom_command(TARGET facesr_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:facesr_train>
        COMMENT "Copying LibTorch DLLs to output directory"
    )

    # 为测试程序复制DLL
    add_custom_command(TARGET facesr_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:facesr_test>
        COMMENT "Copying LibTorch DLLs to test output directory"
    )

    # 复制Qt DLL (仅GUI)
    if(BUILD_GUI)
        add_custom_command(TARGET facesr_gui_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:facesr_gui_app>
            COMMENT "Copying LibTorch DLLs to GUI output directory"
        )
        # 复制 Qt DLLs (根据版本)
        if(QT_VERSION EQUAL 6)
            add_custom_command(TARGET facesr_gui_app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Qt6::Core>
                $<TARGET_FILE:Qt6::Widgets>
                $<TARGET_FILE:Qt6::Gui>
                $<TARGET_FILE_DIR:facesr_gui_app>
                COMMENT "Copying Qt6 DLLs to output directory"
            )
        else()
            add_custom_command(TARGET facesr_gui_app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Qt5::Core>
                $<TARGET_FILE:Qt5::Widgets>
                $<TARGET_FILE:Qt5::Gui>
                $<TARGET_FILE_DIR:facesr_gui_app>
                COMMENT "Copying Qt5 DLLs to output directory"
            )
        endif()

        # 复制 OpenCV DLLs
        file(GLOB OPENCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
        if(OPENCV_DLLS)
            add_custom_command(TARGET facesr_gui_app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_DLLS}
                $<TARGET_FILE_DIR:facesr_gui_app>
                COMMENT "Copying OpenCV DLLs to GUI output directory"
            )
        endif()
    endif()
endif()

# ==================== 安装配置 ====================

install(TARGETS facesr_train facesr_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(BUILD_GUI)
    install(TARGETS facesr_gui_app
        RUNTIME DESTINATION bin
    )
endif()

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装配置文件模板
install(FILES
    ${CMAKE_SOURCE_DIR}/config/train_config.ini
    ${CMAKE_SOURCE_DIR}/config/ui_config.ini
    DESTINATION config
    OPTIONAL
)

# ==================== 测试配置 ====================

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    # 如果有测试，在此添加
endif()

# ==================== 打印配置信息 ====================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "FaceSR C++ Configuration Summary")
message(STATUS "========================================")
message(STATUS "CMake version:     ${CMAKE_VERSION}")
message(STATUS "C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  LibTorch:        ${TORCH_INSTALL_PREFIX}")
message(STATUS "  OpenCV:          ${OpenCV_VERSION}")
if(BUILD_GUI)
    if(QT_VERSION EQUAL 6)
        message(STATUS "  Qt6:             ${Qt6_VERSION}")
    else()
        message(STATUS "  Qt5:             ${Qt5_VERSION}")
    endif()
else()
    message(STATUS "  Qt:              Not found/disabled")
endif()
message(STATUS "")
message(STATUS "CUDA compiler:     ${CUDA_AVAILABLE}")
message(STATUS "CUDA (LibTorch):   ${TORCH_CUDA_VERSION}")
message(STATUS "Hybrid GPU+CPU:    Enabled (auto-detect at runtime)")
message(STATUS "Build GUI:         ${BUILD_GUI}")
message(STATUS "Build tests:       ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")
