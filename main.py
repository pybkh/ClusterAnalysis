"""
植物表型聚类分析系统
主程序入口
"""

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from views import MainWindow


def main():
    """主函数"""
    # 设置高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # 设置默认字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
