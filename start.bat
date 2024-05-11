@echo off
REM 设置窗口标题
title wylbot-元龙机器人微信版
REM 激活环境
call conda activate wylbot
REM 运行 Python 脚本
python listen_ws.py
