# 超分辨率人脸图像重建项目 CMake 构建配置
# FaceSR - GAN-based Face Super-Resolution using C++ (LibTorch + OpenCV + Qt6)

cmake_minimum_required(VERSION 3.18)  # 要求CMake最低版本3.18（LibTorch需要）
project(FaceSR VERSION 1.0.0 LANGUAGES CXX)  # 定义项目名称FaceSR，版本1.0.0，使用C++语言

# ============================================================================
# 编译选项
# ============================================================================

set(CMAKE_CXX_STANDARD 17)          # 使用C++17标准（折叠表达式、结构化绑定等特性）
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求C++17，编译器不支持则报错
set(CMAKE_CXX_EXTENSIONS OFF)       # 禁用编译器特有扩展（如GNU扩展），确保标准合规

# 如果检测到CUDA编译器，设置CUDA也使用C++17标准
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# 如果未指定构建类型（Debug/Release），默认使用Release模式（最大性能优化）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release模式编译优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # MSVC编译器: /O2最大速度优化, /Ob2内联扩展, /DNDEBUG禁用assert断言
        add_compile_options(/O2 /Ob2 /DNDEBUG)
    else()
        # GCC/Clang: -O3最高级优化, -DNDEBUG禁用assert
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# Debug模式: 禁用优化，生成调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        add_compile_options(/Od /Zi)   # /Od禁用优化, /Zi生成PDB调试信息
    else()
        add_compile_options(-O0 -g)    # -O0无优化, -g生成调试符号
    endif()
endif()

# 设置输出目录 — 可执行文件输出到bin/，库文件输出到lib/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# 导出compile_commands.json，供IDE(如VSCode、CLion)使用代码补全和跳转
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Linux 需要显式链接 pthread
find_package(Threads REQUIRED)

# ============================================================================
# 查找依赖库
# ============================================================================

# --- LibTorch ---
# 自动查找LibTorch安装路径（如果用户未手动指定CMAKE_PREFIX_PATH）
if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
    if(WIN32)
        # Windows: 先查 %USERPROFILE%/libtorch，再查 C:/libtorch
        if(EXISTS "$ENV{USERPROFILE}/libtorch")
            list(APPEND CMAKE_PREFIX_PATH "$ENV{USERPROFILE}/libtorch")
        elseif(EXISTS "C:/libtorch")
            list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
        endif()
    else()
        # Linux/Mac: 先查 $HOME/libtorch，再查 /usr/local/libtorch
        if(EXISTS "$ENV{HOME}/libtorch")
            list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
        elseif(EXISTS "/usr/local/libtorch")
            list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")
        endif()
    endif()
endif()

# 查找并加载LibTorch，REQUIRED表示找不到就报错终止
find_package(Torch REQUIRED)
# 将LibTorch的编译标志附加到全局C++编译标志中（如-D_GLIBCXX_USE_CXX11_ABI=0等）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 将protobuf标记为已找到，防止OpenCV重复查找protobuf导致版本冲突
# （LibTorch自带protobuf，OpenCV也可能依赖protobuf，两者版本不同会冲突）
set(protobuf_FOUND TRUE)
set(Protobuf_FOUND TRUE)

# --- OpenCV ---
# 查找OpenCV，需要核心模块(core)、图像处理(imgproc)、图像编解码(imgcodecs)、高级GUI(highgui)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# --- Qt (可选，仅GUI需要) ---
# 启用Qt的自动化工具:
set(CMAKE_AUTOMOC ON)   # 自动运行MOC(元对象编译器)处理Q_OBJECT宏
set(CMAKE_AUTORCC ON)   # 自动处理资源文件(.qrc)
set(CMAKE_AUTOUIC ON)   # 自动处理UI文件(.ui)

# GUI构建开关，默认开启
option(BUILD_GUI "Build GUI application" ON)

if(BUILD_GUI)
    # 优先查找Qt6
    find_package(Qt6 QUIET COMPONENTS Widgets)
    if(Qt6_FOUND)
        set(QT_LIBRARIES Qt6::Widgets)
        message(STATUS "Found Qt6: ${Qt6_VERSION}")
    else()
        # Qt6找不到，尝试Qt5
        find_package(Qt5 QUIET COMPONENTS Widgets)
        if(Qt5_FOUND)
            set(QT_LIBRARIES Qt5::Widgets)
            message(STATUS "Found Qt5: ${Qt5_VERSION}")
        else()
            # 都找不到则关闭GUI构建
            message(WARNING "Qt not found, GUI will not be built")
            set(BUILD_GUI OFF)
        endif()
    endif()
endif()

# ============================================================================
# 源文件定义
# ============================================================================

# 包含头文件搜索路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 公共头文件列表（config, config_parser, logger, random）
set(COMMON_HEADERS
    include/common/config.h
    include/common/config_parser.h
    include/common/logger.h
    include/common/random.h
)

# 模型源文件（生成器、判别器、损失函数、注意力模块）
set(MODEL_SOURCES
    src/models/generator.cpp
    src/models/discriminator.cpp
    src/models/losses.cpp
    src/models/attention.cpp
)

# 工具源文件（数据集加载、图像处理工具、评估指标）
set(UTILS_SOURCES
    src/utils/dataset.cpp
    src/utils/image_utils.cpp
    src/utils/metrics.cpp
)

# GUI源文件和头文件（仅在BUILD_GUI开启时定义）
# 显式添加头文件是为了确保Qt MOC能正确处理Q_OBJECT宏
if(BUILD_GUI)
    set(GUI_SOURCES
        src/gui/main_window.cpp
        src/gui/style_config.cpp
        src/gui/background_widget.cpp
        src/gui/frosted_widget.cpp
    )
    set(GUI_HEADERS
        include/gui/main_window.h
        include/gui/style_config.h
        include/gui/background_widget.h
        include/gui/frosted_widget.h
    )
endif()

# ============================================================================
# 构建目标
# ============================================================================

# --- facesr_models 静态库 --- 包含生成器、判别器、损失函数
add_library(facesr_models STATIC ${MODEL_SOURCES})
target_include_directories(facesr_models PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(facesr_models PUBLIC ${TORCH_LIBRARIES})

# --- facesr_utils 静态库 --- 包含数据集、图像工具、指标计算
add_library(facesr_utils STATIC ${UTILS_SOURCES})
target_include_directories(facesr_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(facesr_utils PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# --- facesr_gui 静态库 (可选) --- GUI组件
if(BUILD_GUI)
    add_library(facesr_gui STATIC ${GUI_SOURCES} ${GUI_HEADERS})
    target_include_directories(facesr_gui PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(facesr_gui PUBLIC
        facesr_models
        facesr_utils
        ${QT_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
    )
    # 设置AUTOMOC=ON确保Q_OBJECT宏被正确处理
    set_target_properties(facesr_gui PROPERTIES AUTOMOC ON)
endif()

# --- facesr_train 训练可执行程序 ---
add_executable(facesr_train
    src/main_train.cpp    # 训练入口
    src/trainer.cpp       # 训练器实现
)
target_include_directories(facesr_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(facesr_train PRIVATE
    facesr_models
    facesr_utils
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)

# --- facesr_test 推理可执行程序 ---
add_executable(facesr_test
    src/main_test.cpp     # 推理入口
    src/inference.cpp     # 推理器实现
)
target_include_directories(facesr_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(facesr_test PRIVATE
    facesr_models
    facesr_utils
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)

# --- facesr_gui_app GUI可执行程序 (可选) ---
if(BUILD_GUI)
    add_executable(facesr_gui_app
        src/main_gui.cpp  # GUI入口
    )
    target_include_directories(facesr_gui_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(facesr_gui_app PRIVATE
        facesr_gui
        facesr_models
        facesr_utils
        ${QT_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
    )
    # WIN32_EXECUTABLE=TRUE 在Windows上隐藏控制台窗口（仅显示GUI窗口）
    set_target_properties(facesr_gui_app PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
endif()

# ============================================================================
# Windows DLL复制
# Windows下动态链接库(DLL)必须和exe在同一目录才能被找到
# ============================================================================
if(WIN32)
    # --- 将LibTorch DLL复制到训练程序输出目录 ---
    if(EXISTS "${TORCH_INSTALL_PREFIX}/lib")
        add_custom_command(TARGET facesr_train POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_INSTALL_PREFIX}/lib"
            "$<TARGET_FILE_DIR:facesr_train>"
            COMMENT "Copying LibTorch DLLs for training executable"
        )
    endif()

    # --- 将LibTorch DLL复制到测试程序输出目录 ---
    if(EXISTS "${TORCH_INSTALL_PREFIX}/lib")
        add_custom_command(TARGET facesr_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_INSTALL_PREFIX}/lib"
            "$<TARGET_FILE_DIR:facesr_test>"
            COMMENT "Copying LibTorch DLLs for test executable"
        )
    endif()

    # --- 将LibTorch/Qt/OpenCV DLL复制到GUI程序输出目录 ---
    if(BUILD_GUI)
        # 复制LibTorch DLL
        if(EXISTS "${TORCH_INSTALL_PREFIX}/lib")
            add_custom_command(TARGET facesr_gui_app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${TORCH_INSTALL_PREFIX}/lib"
                "$<TARGET_FILE_DIR:facesr_gui_app>"
                COMMENT "Copying LibTorch DLLs for GUI executable"
            )
        endif()

        # 复制Qt DLL (使用windeployqt自动查找需要的Qt DLL)
        if(Qt6_FOUND)
            find_program(WINDEPLOYQT windeployqt HINTS "${Qt6_DIR}/../../../bin")
        elseif(Qt5_FOUND)
            find_program(WINDEPLOYQT windeployqt HINTS "${Qt5_DIR}/../../../bin")
        endif()
        if(WINDEPLOYQT)
            add_custom_command(TARGET facesr_gui_app POST_BUILD
                COMMAND ${WINDEPLOYQT} "$<TARGET_FILE:facesr_gui_app>"
                COMMENT "Running windeployqt to copy Qt DLLs"
            )
        endif()

        # 复制OpenCV DLL
        foreach(opencv_lib ${OpenCV_LIBS})
            if(TARGET ${opencv_lib})
                add_custom_command(TARGET facesr_gui_app POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${opencv_lib}>"
                    "$<TARGET_FILE_DIR:facesr_gui_app>"
                    COMMENT "Copying ${opencv_lib} DLL for GUI executable"
                )
            endif()
        endforeach()
    endif()
endif()

# ============================================================================
# 安装规则
# ============================================================================

# 安装可执行文件到 <prefix>/bin
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

# 安装头文件到 <prefix>/include
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装配置文件模板
install(DIRECTORY config/
    DESTINATION share/facesr/config
    FILES_MATCHING
    PATTERN "*.ini"
    PATTERN "*.yaml"
)

# ============================================================================
# 测试 (默认关闭)
# ============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # 预留测试子目录
endif()

# ============================================================================
# 打印配置摘要信息
# ============================================================================
message(STATUS "")
message(STATUS "=== FaceSR Build Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "  LibTorch:       ${Torch_DIR}")
message(STATUS "  CUDA:           ${TORCH_CUDA_LIBRARIES}")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
if(BUILD_GUI)
    if(Qt6_FOUND)
        message(STATUS "  Qt:             Qt6 ${Qt6_VERSION}")
    elseif(Qt5_FOUND)
        message(STATUS "  Qt:             Qt5 ${Qt5_VERSION}")
    endif()
else()
    message(STATUS "  Qt:             Not found (GUI disabled)")
endif()
message(STATUS "")
message(STATUS "  Build GUI:      ${BUILD_GUI}")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "=================================")
message(STATUS "")
