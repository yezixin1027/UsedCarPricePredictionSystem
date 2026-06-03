# src/config.py — 向后兼容的重导出模块
# 项目配置已提升到根目录 config.py，此文件仅作转发
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *  # noqa: F401 F403
