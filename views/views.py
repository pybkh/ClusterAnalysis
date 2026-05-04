"""
视图层 (View)
负责 GUI 界面的构建和用户交互
"""

import sys
import os
from typing import Optional, List

# 设置 matplotlib 使用 PySide6 后端（必须在导入 matplotlib 之前）
os.environ['QT_API'] = 'pyside6'
os.environ['MPLBACKEND'] = 'QtAgg'

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox,
    QSpinBox, QRadioButton, QButtonGroup, QGroupBox, QTableView,
    QProgressDialog, QMessageBox, QHeaderView, QAbstractItemView,
    QStyledItemDelegate, QApplication, QInputDialog
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, Slot, QSize
from PySide6.QtGui import QColor, QFont, QIcon

from models import DataManager
from controllers import ClusteringWorker
from models.utils import get_display_name, is_feature_column, FEATURE_COLUMNS


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SqlTableModel(QAbstractTableModel):
    """
    基于 SQLite 的表格模型
    用于驱动 QTableView 显示数据
    """

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self._data_manager = data_manager
        self._data: Optional[pd.DataFrame] = None
        self._refresh_data()

    def _refresh_data(self):
        """从数据库刷新数据"""
        self.beginResetModel()
        self._data = self._data_manager.get_all_data()
        self.endResetModel()

    def refresh(self):
        """外部调用刷新"""
        self._refresh_data()

    def rowCount(self, parent=QModelIndex()) -> int:
        if self._data is None:
            return 0
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        if self._data is None:
            return 0
        return len(self._data.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or self._data is None:
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        if role == Qt.BackgroundRole:
            # 根据 cluster_id 设置行背景色
            if 'cluster_id' in self._data.columns:
                cluster_id = self._data.iloc[index.row()]['cluster_id']
                if pd.notna(cluster_id):
                    colors = [
                        QColor(230, 240, 255),  # 浅蓝
                        QColor(255, 230, 230),  # 浅红
                        QColor(230, 255, 230),  # 浅绿
                        QColor(255, 255, 230),  # 浅黄
                        QColor(255, 230, 255),  # 浅紫
                        QColor(230, 255, 255),  # 浅青
                    ]
                    return colors[int(cluster_id) % len(colors)]

        return None

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal and self._data is not None:
            col_name = self._data.columns[section]
            return get_display_name(col_name)
        elif orientation == Qt.Vertical:
            return str(section + 1)

        return None

    def get_column_name(self, col: int) -> str:
        """获取原始列名"""
        if self._data is not None and col < len(self._data.columns):
            return self._data.columns[col]
        return ""

    def get_row_db_index(self, row: int) -> Optional[int]:
        """获取指定行的数据库索引"""
        if self._data is not None and 'original_db_index' in self._data.columns:
            if row < len(self._data):
                return int(self._data.iloc[row]['original_db_index'])
        return None


class ScatterCanvas(FigureCanvas):
    """
    Matplotlib 散点图画布
    嵌入 PySide6 窗口，用于可视化聚类结果
    """

    # 点击散点时发射信号，参数为 original_db_index
    point_clicked = Signal(int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self._scatter = None
        self._db_indices = None
        self._highlight_scatter = None

        # 连接 pick 事件
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)

    def plot_clusters(self, df: pd.DataFrame, feature_x: str, feature_y: str,
                      cluster_col: str = 'cluster_id', sample_size: int = 5000):
        """
        绘制聚类散点图

        Args:
            df: 包含特征和 cluster_id 的 DataFrame
            feature_x: X 轴特征列名
            feature_y: Y 轴特征列名
            cluster_col: 聚类 ID 列名
            sample_size: 最大采样点数
        """
        self.axes.clear()

        # 大数据集自动采样
        if len(df) > sample_size:
            plot_df = df.sample(n=sample_size, random_state=42)
        else:
            plot_df = df.copy()

        # 获取数据
        x = plot_df[feature_x].values
        y = plot_df[feature_y].values
        clusters = plot_df[cluster_col].values if cluster_col in plot_df.columns else None
        self._db_indices = plot_df['original_db_index'].values if 'original_db_index' in plot_df.columns else None

        # 颜色映射
        if clusters is not None:
            unique_clusters = sorted(set(clusters[~pd.isna(clusters)]))
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
            color_map = {c: colors[i] for i, c in enumerate(unique_clusters)}
            point_colors = [color_map.get(c, 'gray') if pd.notna(c) else 'gray' for c in clusters]
        else:
            point_colors = 'steelblue'
            unique_clusters = []

        # 绘制散点图
        self._scatter = self.axes.scatter(
            x, y, c=point_colors, alpha=0.6, s=30,
            picker=True, pickradius=5
        )

        # 设置坐标轴标签（中文）
        self.axes.set_xlabel(get_display_name(feature_x), fontsize=12)
        self.axes.set_ylabel(get_display_name(feature_y), fontsize=12)
        self.axes.set_title('聚类结果散点图', fontsize=14)

        # 添加图例
        if clusters is not None and len(unique_clusters) > 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color_map[c], label=f'类别 {int(c)}')
                for c in unique_clusters
            ]
            self.axes.legend(handles=legend_elements, loc='best')

        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()

    def highlight_point(self, db_index: int):
        """
        高亮指定的点

        Args:
            db_index: 数据库索引
        """
        # 移除旧的高亮
        if self._highlight_scatter is not None:
            try:
                self._highlight_scatter.remove()
            except (NotImplementedError, ValueError):
                pass  # 忽略移除失败的情况
            self._highlight_scatter = None

        if self._db_indices is None or self._scatter is None:
            return

        # 找到对应的点
        if db_index in self._db_indices:
            idx = np.where(self._db_indices == db_index)[0]
            if len(idx) > 0:
                offsets = self._scatter.get_offsets()
                point = offsets[idx[0]]

                # 绘制高亮点
                self._highlight_scatter = self.axes.scatter(
                    [point[0]], [point[1]],
                    s=200, facecolors='none', edgecolors='red',
                    linewidths=2, zorder=10
                )
                self.draw()

    def _on_pick(self, event):
        """处理点击事件"""
        if event.artist != self._scatter:
            return

        ind = event.ind
        if len(ind) > 0 and self._db_indices is not None:
            idx = ind[0]
            db_index = int(self._db_indices[idx])
            self.point_clicked.emit(db_index)


class MainWindow(QMainWindow):
    """
    主窗口类
    实现完整的 GUI 界面
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("植物表型聚类分析系统")
        self.setMinimumSize(1200, 800)

        # 初始化数据管理器
        self._data_manager = DataManager()
        self._worker: Optional[ClusteringWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None
        self._raw_df: Optional[pd.DataFrame] = None  # 临时存储原始数据

        # 初始化表格模型
        self._table_model = SqlTableModel(self._data_manager)

        # 创建 UI
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """构建用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部工具栏
        toolbar_layout = QHBoxLayout()
        self._btn_open = QPushButton("打开文件")
        self._btn_export = QPushButton("导出结果")
        self._btn_export.setEnabled(False)
        self._lbl_status = QLabel("就绪")
        toolbar_layout.addWidget(self._btn_open)
        toolbar_layout.addWidget(self._btn_export)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self._lbl_status)
        main_layout.addLayout(toolbar_layout)

        # 主分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 参数配置组
        self._config_group = QGroupBox("参数配置")
        config_layout = QVBoxLayout(self._config_group)

        # 特征选择
        feature_label = QLabel("选择特征列（至少2个）:")
        config_layout.addWidget(feature_label)

        self._feature_checkboxes_layout = QVBoxLayout()
        self._feature_checkboxes_widget = QWidget()
        self._feature_checkboxes_widget.setLayout(self._feature_checkboxes_layout)
        config_layout.addWidget(self._feature_checkboxes_widget)

        # K 值设置
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("聚类数 K:"))
        self._spin_k = QSpinBox()
        self._spin_k.setMinimum(2)
        self._spin_k.setMaximum(20)
        self._spin_k.setValue(3)
        k_layout.addWidget(self._spin_k)
        config_layout.addLayout(k_layout)

        self._radio_auto_k = QRadioButton("自动优化 K 值")
        config_layout.addWidget(self._radio_auto_k)

        # 运行按钮
        self._btn_run = QPushButton("运行聚类")
        self._btn_run.setEnabled(False)
        config_layout.addWidget(self._btn_run)

        config_layout.addStretch()
        left_layout.addWidget(self._config_group)

        # 可视化配置组
        self._viz_group = QGroupBox("可视化分析")
        viz_layout = QVBoxLayout(self._viz_group)

        viz_layout.addWidget(QLabel("X 轴特征:"))
        self._combo_x = QComboBox()
        viz_layout.addWidget(self._combo_x)

        viz_layout.addWidget(QLabel("Y 轴特征:"))
        self._combo_y = QComboBox()
        viz_layout.addWidget(self._combo_y)

        self._btn_plot = QPushButton("更新图表")
        self._btn_plot.setEnabled(False)
        viz_layout.addWidget(self._btn_plot)

        viz_layout.addStretch()
        self._viz_group.setVisible(False)
        left_layout.addWidget(self._viz_group)

        splitter.addWidget(left_panel)

        # 右侧区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 表格视图
        self._table_view = QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table_view.verticalHeader().setDefaultSectionSize(25)
        right_layout.addWidget(self._table_view, stretch=1)

        # Matplotlib 画布
        self._canvas = ScatterCanvas()
        self._canvas.setMinimumHeight(400)
        right_layout.addWidget(self._canvas, stretch=2)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])

    def _connect_signals(self):
        """连接信号和槽"""
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_run.clicked.connect(self._on_run_clustering)
        self._btn_plot.clicked.connect(self._on_plot_update)

        # K 值设置联动
        self._radio_auto_k.toggled.connect(self._spin_k.setDisabled)

        # 表格选择变化
        self._table_view.selectionModel().selectionChanged.connect(self._on_table_selection_changed)

        # 散点图点击
        self._canvas.point_clicked.connect(self._on_point_clicked)

    def _on_open_file(self):
        """打开文件按钮点击处理"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "数据文件 (*.csv *.xlsx *.xls);;所有文件 (*)"
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()

            # Excel 文件选择 sheet
            sheet_name = None
            if ext in ['.xlsx', '.xls']:
                sheets = self._data_manager.get_excel_sheets(file_path)
                if not sheets:
                    QMessageBox.warning(self, "警告", "无法读取 Excel 工作表")
                    return
                if len(sheets) > 1:
                    sheet_name, ok = QComboBox.getItem(
                        self, "选择工作表", "请选择要导入的工作表:",
                        sheets, 0, False
                    )
                    if not ok:
                        return

            # 加载文件
            result, raw_df = self._data_manager.load_file(file_path, sheet_name)

            if result is False and raw_df is not None:
                # 需要手动选择表头
                self._raw_df = raw_df
                self._show_header_selection_dialog(raw_df)
                return

            # 加载成功
            self._on_data_loaded()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件失败:\n{str(e)}")

    def _show_header_selection_dialog(self, raw_df: pd.DataFrame):
        """显示表头选择对话框"""
        from PySide6.QtWidgets import QInputDialog

        # 预览前 10 行
        preview = raw_df.head(10).to_string()
        msg = f"未能自动识别表头行。\n\n前 10 行数据预览:\n{preview}\n\n请输入表头行号（从 0 开始）:"

        row, ok = QInputDialog.getInt(
            self, "手动选择表头", msg, 0, 0, len(raw_df) - 1, 1
        )

        if ok:
            try:
                self._data_manager.load_with_manual_header(self._raw_df, row)
                self._raw_df = None
                self._on_data_loaded()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")

    def _on_data_loaded(self):
        """数据加载完成后的处理"""
        # 刷新表格
        self._table_model.refresh()

        # 更新状态
        row_count = self._data_manager.get_row_count()
        self._lbl_status.setText(f"已加载 {row_count} 条数据")

        # 生成特征选择框
        self._create_feature_checkboxes()

        # 启用导出按钮
        self._btn_export.setEnabled(True)

        # 重置可视化
        self._viz_group.setVisible(False)
        self._btn_plot.setEnabled(False)

    def _create_feature_checkboxes(self):
        """动态生成特征列勾选框"""
        # 清除旧的复选框
        while self._feature_checkboxes_layout.count():
            item = self._feature_checkboxes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 根据数据中的特征列生成复选框
        self._feature_checkboxes = []
        for col in self._data_manager.feature_columns:
            cb = QCheckBox(get_display_name(col))
            cb.setProperty('column_name', col)
            cb.toggled.connect(self._on_feature_toggled)
            self._feature_checkboxes_layout.addWidget(cb)
            self._feature_checkboxes.append(cb)

    def _on_feature_toggled(self):
        """特征选择变化处理"""
        checked_count = sum(1 for cb in self._feature_checkboxes if cb.isChecked())
        self._btn_run.setEnabled(checked_count >= 2)

    def _get_selected_features(self) -> List[str]:
        """获取用户选择的特征列"""
        return [
            cb.property('column_name')
            for cb in self._feature_checkboxes
            if cb.isChecked()
        ]

    def _on_run_clustering(self):
        """运行聚类按钮点击处理"""
        selected_features = self._get_selected_features()
        if len(selected_features) < 2:
            QMessageBox.warning(self, "警告", "请至少选择 2 个特征列")
            return

        # 获取聚类数据
        df = self._data_manager.get_data_for_clustering(selected_features)
        if df is None or df.empty:
            QMessageBox.warning(self, "警告", "无法获取数据")
            return

        # 确定 K 值
        n_clusters = None if self._radio_auto_k.isChecked() else self._spin_k.value()

        # 创建进度对话框
        self._progress_dialog = QProgressDialog("正在计算聚类...", "取消", 0, 0, self)
        self._progress_dialog.setWindowTitle("计算中")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        # 创建并启动工作线程
        self._worker = ClusteringWorker(df, selected_features, n_clusters)
        self._worker.progress_signal.connect(self._on_worker_progress)
        self._worker.result_signal.connect(self._on_worker_result)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker.finished_signal.connect(self._on_worker_finished)

        self._progress_dialog.canceled.connect(self._worker.cancel)
        self._worker.start()

        # 禁用按钮
        self._btn_run.setEnabled(False)
        self._btn_open.setEnabled(False)

    @Slot(str)
    def _on_worker_progress(self, message: str):
        """更新进度信息"""
        if self._progress_dialog:
            self._progress_dialog.setLabelText(message)

    @Slot(object, object)
    def _on_worker_result(self, result_df: pd.DataFrame, centroids_df: pd.DataFrame):
        """处理聚类结果"""
        # 更新数据库
        self._data_manager.update_cluster_ids(result_df)

        # 刷新表格
        self._table_model.refresh()

        # 显示可视化配置
        self._viz_group.setVisible(True)
        self._update_viz_comboboxes()

        # 自动绘图
        self._auto_plot(result_df)

        # 显示聚类中心信息
        self._show_centroids_info(centroids_df)

    @Slot(str)
    def _on_worker_error(self, error_msg: str):
        """处理错误"""
        QMessageBox.warning(self, "聚类警告", error_msg)

    @Slot()
    def _on_worker_finished(self):
        """工作线程完成"""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        self._btn_run.setEnabled(True)
        self._btn_open.setEnabled(True)

    def _update_viz_comboboxes(self):
        """更新可视化下拉框"""
        self._combo_x.clear()
        self._combo_y.clear()

        features = self._get_selected_features()
        for col in features:
            display_name = get_display_name(col)
            self._combo_x.addItem(display_name, col)
            self._combo_y.addItem(display_name, col)

        # 默认选择前两个不同的特征
        if len(features) >= 2:
            self._combo_x.setCurrentIndex(0)
            self._combo_y.setCurrentIndex(1)
            self._btn_plot.setEnabled(True)

    def _auto_plot(self, result_df: pd.DataFrame):
        """自动绘制聚类结果图"""
        df = self._data_manager.get_all_data()
        if df is None:
            return

        features = self._get_selected_features()
        if len(features) >= 2:
            self._canvas.plot_clusters(df, features[0], features[1])

    def _on_plot_update(self):
        """更新图表按钮点击"""
        feature_x = self._combo_x.currentData()
        feature_y = self._combo_y.currentData()

        if feature_x and feature_y:
            df = self._data_manager.get_all_data()
            if df is not None:
                self._canvas.plot_clusters(df, feature_x, feature_y)

    def _show_centroids_info(self, centroids_df: pd.DataFrame):
        """显示聚类中心信息"""
        if centroids_df is None or centroids_df.empty:
            return

        msg = "聚类中心特征值:\n\n"
        for idx, row in centroids_df.iterrows():
            msg += f"类别 {idx}:\n"
            for col in centroids_df.columns:
                msg += f"  {get_display_name(col)}: {row[col]:.4f}\n"
            msg += "\n"

        QMessageBox.information(self, "聚类中心", msg)

    @Slot()
    def _on_table_selection_changed(self):
        """表格选择变化处理"""
        indexes = self._table_view.selectionModel().selectedRows()
        if not indexes:
            return

        row = indexes[0].row()
        db_index = self._table_model.get_row_db_index(row)
        if db_index is not None:
            self._canvas.highlight_point(db_index)

    @Slot(int)
    def _on_point_clicked(self, db_index: int):
        """散点图点击处理"""
        # 查找对应的表格行
        model = self._table_model
        for row in range(model.rowCount()):
            if model.get_row_db_index(row) == db_index:
                self._table_view.selectRow(row)
                self._table_view.scrollTo(model.index(row, 0))
                break

    def _on_export(self):
        """导出结果"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "聚类结果.xlsx",
            "Excel 文件 (*.xlsx);;所有文件 (*)"
        )
        if not file_path:
            return

        try:
            self._data_manager.export_to_excel(file_path)
            QMessageBox.information(self, "成功", f"结果已导出到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self._data_manager.close()
        event.accept()
