#!/bin/bash

# AutoGPT 项目环境设置脚本
#
# 本脚本负责为 AutoGPT 项目设置必要的开发环境，包括 Python 和 Poetry 的安装。
# 支持 Linux 和 MacOS 系统，Windows 用户需要使用 WSL 或手动安装。
#
# 主要功能:
#   - 检测操作系统类型
#   - 自动安装 Python 3.11.5（通过 pyenv）
#   - 自动安装 Poetry 包管理器
#   - 提供 Windows 用户的安装指导
#
# 使用方法:
#   chmod +x setup.sh
#   ./setup.sh
#
# 依赖工具:
#   - curl: 用于下载安装脚本
#   - bash: 脚本执行环境

# 检查操作系统类型，Windows 系统不支持此脚本
if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "This script cannot be run on Windows."
    echo "Please follow the installation instructions at https://docs.python.org/3/using/windows.html"
    echo "To install poetry on Windows, please follow the instructions at https://python-poetry.org/docs/master/#installation"
    
    exit 1
else
    # 检查并安装 Python 3
    if ! command -v python3 &> /dev/null
    then
        echo "python3 could not be found"
        echo "Installing python3 using pyenv..."
        
        # 检查并安装 pyenv（Python 版本管理工具）
        if ! command -v pyenv &> /dev/null
        then
            echo "pyenv could not be found"
            echo "Installing pyenv..."
            # 使用官方安装脚本安装 pyenv
            curl https://pyenv.run | bash
        fi
        
        # 安装 Python 3.11.5 并设置为全局默认版本
        pyenv install 3.11.5
        pyenv global 3.11.5
    fi

    # 检查并安装 Poetry（Python 包管理和构建工具）
    if ! command -v poetry &> /dev/null
    then
        echo "poetry could not be found"
        echo "Installing poetry..."
        # 使用官方安装脚本安装 Poetry
        curl -sSL https://install.python-poetry.org | python3 -
    fi
fi
