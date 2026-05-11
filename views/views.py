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
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox,
    QSpinBox, QRadioButton, QButtonGroup, QGroupBox, QTableView,
    QProgressDialog, QMessageBox, QHeaderView, QAbstractItemView,
    QStyledItemDelegate, QApplication, QInputDialog
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, Slot, QSize, QThread
from PySide6.QtGui import QColor, QFont, QIcon

from models import DataManager
from controllers import ClusteringWorker
from models.analyzer import ClusteringParams, ClusteringResult
from models.utils import get_display_name, is_feature_column, FEATURE_COLUMNS


class SqlTableModel(QAbstractTableModel):
    """
    基于 SQLite 的表格模型
    用于驱动 QTableView 显示数据
    支持按品种过滤
    """

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self._data_manager = data_manager
        self._data: Optional[pd.DataFrame] = None
        self._variety_filter: Optional[str] = None  # None=全部, str=品种名
        self._refresh_data()

    def set_variety_filter(self, variety_name: Optional[str] = None):
        """
        设置品种过滤条件

        Args:
            variety_name: 品种名称，None 表示显示所有数据
        """
        self._variety_filter = variety_name
        self._refresh_data()

    def get_variety_filter(self) -> Optional[str]:
        """获取当前品种过滤条件"""
        return self._variety_filter

    def _refresh_data(self):
        """从数据库刷新数据（按当前品种过滤）"""
        self.beginResetModel()
        if self._variety_filter is None:
            self._data = self._data_manager.get_all_data()
        else:
            self._data = self._data_manager.get_filtered_data(self._variety_filter)
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
            col_name = self._data.columns[index.column()]
            row_data = self._data.iloc[index.row()]
            value = row_data[col_name]

            if pd.isna(value):
                return ""

            # 全部品种模式下，cluster_id 显示为 "品种名_簇号"（需求9.4）
            if col_name == 'cluster_id' and self._variety_filter is None:
                variety = row_data.get('variety_name', '')
                if pd.notna(variety) and pd.notna(value):
                    return f"{variety}_{int(value)}"

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

    支持 PCA 降维至 2 维绘制散点图
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

        # 设置中文字体（仅对当前 Figure 生效，不修改全局 rcParams）
        self._chinese_font = font_manager.FontProperties(family='SimHei')

        # 连接 pick 事件
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)

    def plot_clusters(self, df: pd.DataFrame, features: List[str],
                      cluster_col: str = 'cluster_id', sample_size: int = 5000,
                      use_pca: bool = True, color_by: str = 'cluster_id',
                      variety_name: Optional[str] = None,
                      n_clusters: Optional[int] = None):
        """
        绘制聚类散点图

        Args:
            df: 包含特征和 cluster_id 的 DataFrame
            features: 特征列名列表
            cluster_col: 聚类 ID 列名
            sample_size: 最大采样点数
            use_pca: 是否使用 PCA 降维（当特征数 > 2 时）
            color_by: 着色方式 — 'cluster_id' 或 'variety_name'
            variety_name: 当前品种名（用于标题），None 表示全部品种
            n_clusters: K 值（用于标题）
        """
        self.axes.clear()

        # 大数据集自动采样
        if len(df) > sample_size:
            plot_df = df.sample(n=sample_size, random_state=42)
        else:
            plot_df = df.copy()

        # 获取数据
        self._db_indices = plot_df['original_db_index'].values if 'original_db_index' in plot_df.columns else None

        # 确定绘图数据
        if use_pca and len(features) > 2:
            pca = PCA(n_components=2)
            feature_data = plot_df[features].dropna()
            if len(feature_data) < 2:
                self.axes.text(0.5, 0.5, '数据不足，无法进行 PCA',
                             ha='center', va='center', fontproperties=self._chinese_font)
                self.draw()
                return
            pca_result = pca.fit_transform(feature_data)
            x = pca_result[:, 0]
            y = pca_result[:, 1]
            x_label = '主成分 1'
            y_label = '主成分 2'
        else:
            x = plot_df[features[0]].values if len(features) > 0 else np.zeros(len(plot_df))
            y = plot_df[features[1]].values if len(features) > 1 else np.zeros(len(plot_df))
            x_label = get_display_name(features[0]) if len(features) > 0 else ''
            y_label = get_display_name(features[1]) if len(features) > 1 else ''

        # --- 着色逻辑 ---
        from matplotlib.patches import Patch

        if color_by == 'variety_name' and 'variety_name' in plot_df.columns:
            # 按品种名称着色（全部品种模式）
            unique_varieties = sorted(plot_df['variety_name'].dropna().unique())
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_varieties)))
            color_map = {v: colors[i] for i, v in enumerate(unique_varieties)}
            point_colors = [color_map.get(v, 'gray')
                           for v in plot_df['variety_name'].values]

            # 图例
            legend_elements = [
                Patch(facecolor=color_map[v], label=v)
                for v in unique_varieties
            ]
            # 标题
            if use_pca and len(features) > 2:
                title = f'品种分布图（{len(unique_varieties)}个品种）'
            else:
                title = f'品种分布图（{len(unique_varieties)}个品种）'
        else:
            # 按 cluster_id 着色（单品种模式）
            clusters = plot_df[cluster_col].values if cluster_col in plot_df.columns else None
            if clusters is not None:
                valid_mask = ~pd.isna(clusters)
                unique_clusters = sorted(set(clusters[valid_mask]))
                colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
                color_map = {c: colors[i] for i, c in enumerate(unique_clusters)}
                point_colors = [color_map.get(c, 'gray') if pd.notna(c) else 'gray' for c in clusters]

                # 图例
                legend_elements = [
                    Patch(facecolor=color_map[c], label=f'类别 {int(c)}')
                    for c in unique_clusters
                ]
            else:
                point_colors = 'steelblue'
                legend_elements = []

            # 标题
            if use_pca and len(features) > 2:
                if variety_name:
                    k_info = f' (K={n_clusters})' if n_clusters else ''
                    title = f'{variety_name} - PCA聚类结果{k_info}'
                else:
                    title = 'PCA 降维聚类结果'
            else:
                if variety_name:
                    k_info = f' (K={n_clusters})' if n_clusters else ''
                    title = f'{variety_name} - 聚类结果{k_info}'
                else:
                    title = '聚类结果散点图'

        # 绘制散点图
        self._scatter = self.axes.scatter(
            x, y, c=point_colors, alpha=0.6, s=30,
            picker=True, pickradius=5
        )

        # 设置坐标轴标签
        self.axes.set_xlabel(x_label, fontsize=12, fontproperties=self._chinese_font)
        self.axes.set_ylabel(y_label, fontsize=12, fontproperties=self._chinese_font)
        self.axes.set_title(title, fontsize=14, fontproperties=self._chinese_font)

        # 添加图例
        if legend_elements:
            self.axes.legend(handles=legend_elements, loc='best', prop=self._chinese_font)

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
                pass
            self._highlight_scatter = None

        if self._db_indices is None or self._scatter is None:
            return

        # 找到对应的点
        if db_index in self._db_indices:
            idx = np.where(self._db_indices == db_index)[0]
            if len(idx) > 0:
                offsets = self._scatter.get_offsets()
                point = offsets[idx[0]]

                # 绘制高亮点（marker 放大、加粗边框）
                self._highlight_scatter = self.axes.scatter(
                    [point[0]], [point[1]],
                    s=200, facecolors='none', edgecolors='red',
                    linewidths=3, zorder=10
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
        self.setMinimumSize(1280, 900)

        # 初始化数据管理器
        self._data_manager = DataManager()
        self._worker_thread: Optional[QThread] = None
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
        self._chk_chinese_headers = QCheckBox("导出中文表头")
        self._chk_split_by_group = QCheckBox("按品种拆分Sheet")
        self._lbl_status = QLabel("就绪")
        toolbar_layout.addWidget(self._btn_open)
        toolbar_layout.addWidget(self._btn_export)
        toolbar_layout.addWidget(self._chk_chinese_headers)
        toolbar_layout.addWidget(self._chk_split_by_group)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self._lbl_status)
        main_layout.addLayout(toolbar_layout)

        # 主分割器（四等分布局：左侧配置面板 + 右侧上表格下画布）
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 品种信息标签
        self._lbl_group_info = QLabel("品种数量: 0")
        left_layout.addWidget(self._lbl_group_info)

        # 品种选择（始终可见，放在左侧面板顶部）
        variety_select_layout = QHBoxLayout()
        variety_select_layout.addWidget(QLabel("选择品种:"))
        self._combo_group = QComboBox()
        variety_select_layout.addWidget(self._combo_group, 1)
        left_layout.addLayout(variety_select_layout)

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

        # 全品种聚类复选框
        self._chk_merge_all = QCheckBox("合并所有品种聚类（不推荐）")
        config_layout.addWidget(self._chk_merge_all)

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

        self._chk_pca = QCheckBox("使用 PCA 降维")
        self._chk_pca.setChecked(True)
        viz_layout.addWidget(self._chk_pca)

        self._btn_plot = QPushButton("更新图表")
        self._btn_plot.setEnabled(False)
        viz_layout.addWidget(self._btn_plot)

        viz_layout.addStretch()
        self._viz_group.setVisible(False)
        left_layout.addWidget(self._viz_group)

        main_splitter.addWidget(left_panel)

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

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([300, 900])

    def _connect_signals(self):
        """连接信号和槽"""
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_run.clicked.connect(self._on_run_clustering)
        self._btn_plot.clicked.connect(self._on_plot_update)

        # K 值设置联动
        self._radio_auto_k.toggled.connect(self._spin_k.setDisabled)

        # 品种选择变化
        self._combo_group.currentIndexChanged.connect(self._on_group_changed)

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

            # Excel 文件处理
            if ext in ['.xlsx', '.xls']:
                sheets = self._data_manager.get_excel_sheets(file_path)
                if not sheets:
                    QMessageBox.warning(self, "警告", "无法读取 Excel 工作表")
                    return

                if len(sheets) > 1:
                    # 多个工作表：弹出选择对话框
                    sheet_name, merge_all = self._show_sheet_selection_dialog(sheets)
                    if sheet_name is None and not merge_all:
                        return  # 用户取消

                    if merge_all:
                        result = self._data_manager.load_file(file_path, merge_all_sheets=True)
                    else:
                        result = self._data_manager.load_file(file_path, sheet_name)
                else:
                    # 单个工作表：直接加载
                    result = self._data_manager.load_file(file_path, sheets[0])
            else:
                # CSV 文件
                result = self._data_manager.load_file(file_path)

            if not result.success:
                if result.raw_df is not None:
                    # 需要手动选择表头
                    self._raw_df = result.raw_df
                    self._show_header_selection_dialog(result.raw_df)
                    return
                else:
                    QMessageBox.warning(self, "警告", result.error_msg or "加载失败")
                    return

            # 加载成功
            self._on_data_loaded(result.stats)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件失败:\n{str(e)}")

    def _show_sheet_selection_dialog(self, sheets: List[str]) -> tuple:
        """
        显示工作表选择对话框

        Returns:
            tuple: (sheet_name, merge_all) - 用户选择的工作表名和是否合并
        """
        items = sheets + ["合并所有工作表"]
        item, ok = QInputDialog.getItem(
            self, "选择工作表", "请选择要导入的工作表:",
            items, 0, False
        )

        if not ok:
            return None, False

        if item == "合并所有工作表":
            return None, True

        return item, False

    def _show_header_selection_dialog(self, raw_df: pd.DataFrame):
        """显示表头选择对话框"""
        # 预览前 10 行
        preview = raw_df.head(10).to_string()
        msg = f"未能自动识别表头行。\n\n前 10 行数据预览:\n{preview}\n\n请输入表头行号（从 0 开始）:"

        row, ok = QInputDialog.getInt(
            self, "手动选择表头", msg, 0, 0, len(raw_df) - 1, 1
        )

        if ok:
            try:
                result = self._data_manager.load_with_manual_header(self._raw_df, row)
                self._raw_df = None
                if result.success:
                    self._on_data_loaded(result.stats)
                else:
                    QMessageBox.warning(self, "警告", result.error_msg or "加载失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")

    def _on_data_loaded(self, stats: Optional[dict] = None):
        """
        数据加载完成后的处理

        Args:
            stats: 加载统计信息
        """
        # 刷新表格
        self._table_model.refresh()

        # 获取品种信息
        variety_groups = self._data_manager.get_source_groups()
        variety_counts = self._data_manager.get_variety_sample_counts()

        # 更新状态栏品种信息（需求9.7）
        if variety_counts:
            variety_summary = "、".join([
                f"{name}({count}条)"
                for name, count in sorted(variety_counts.items())
            ])
            status_text = f"已加载 {len(variety_counts)} 个品种：{variety_summary}"
        else:
            row_count = self._data_manager.get_row_count()
            status_text = f"已加载 {row_count} 条数据"

        if stats:
            if stats.get('block_count', 1) > 1:
                status_text += f" | {stats['block_count']}个数据块"
            if stats.get('discarded_rows', 0) > 0:
                status_text += f" | 丢弃: {stats['discarded_rows']} 行"

        self._lbl_status.setText(status_text)

        # 更新品种信息标签
        self._lbl_group_info.setText(f"品种数量: {len(variety_groups)}")

        # 处理未知品种：弹窗让用户手动输入（需求9.1.1第4条）
        unknown_varieties = [v for v in variety_groups if '未知品种' in v]
        if unknown_varieties:
            for unknown_name in unknown_varieties:
                new_name, ok = QInputDialog.getText(
                    self, "品种名称确认",
                    f"检测到未识别的品种「{unknown_name}」，\n"
                    f"请输入正确的品种名称：",
                    text=""
                )
                if ok and new_name.strip():
                    try:
                        self._data_manager.rename_variety(unknown_name, new_name.strip())
                    except Exception:
                        pass

            # 重新获取品种列表
            variety_groups = self._data_manager.get_source_groups()
            variety_counts = self._data_manager.get_variety_sample_counts()
            # 刷新表格
            self._table_model.refresh()

        # 填充品种选择下拉框
        self._update_group_combobox(variety_groups, variety_counts)

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

    def _on_feature_toggled(self, checked: bool = False):
        """特征选择变化处理

        Args:
            checked: 复选框状态（PySide6 QCheckBox.toggled 信号携带的参数）
        """
        group_name = self._combo_group.currentData()
        if group_name is None:
            # 全部品种模式 → 始终禁用聚类
            self._btn_run.setEnabled(False)
            return
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
        """运行聚类按钮点击处理（需求9.3 聚类执行模式）"""
        selected_features = self._get_selected_features()
        if len(selected_features) < 2:
            QMessageBox.warning(self, "警告", "请至少选择 2 个特征列")
            return

        current_variety = self._combo_group.currentData()
        merge_all = self._chk_merge_all.isChecked()

        # 模式1：全品种合并聚类（需求9.3.2）
        if merge_all:
            reply = QMessageBox.warning(
                self, "确认合并聚类",
                "⚠️ 警告：合并不同品种进行聚类可能导致生物学意义丧失。\n\n"
                "品种间天然差异（如油菜面积远大于水稻）会主导聚类结果，\n"
                "品种内的表型分化将被掩盖。建议按品种分别聚类分析。",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return

            # 获取所有品种数据
            df = self._data_manager.get_data_for_clustering(selected_features)
            if df is None or df.empty:
                QMessageBox.warning(self, "警告", "无法获取数据")
                return

            variety_groups = self._data_manager.get_source_groups()
            if not variety_groups or variety_groups == ['未知品种']:
                variety_groups = ['全部品种']

            params = ClusteringParams(
                features=selected_features,
                n_clusters=None if self._radio_auto_k.isChecked() else self._spin_k.value(),
                max_k=10
            )

            self._start_clustering_worker(df, params, variety_groups, merge_all=True)
            return

        # 模式2：单品种聚类（需求9.3.1，默认强制）
        if current_variety is None:
            QMessageBox.warning(self, "提示", "请选择单一品种进行聚类")
            return

        # 仅获取当前品种数据
        df = self._data_manager.get_data_by_group(selected_features, current_variety)
        if df is None or df.empty:
            QMessageBox.warning(self, "警告", f"品种「{current_variety}」无有效数据")
            return

        params = ClusteringParams(
            features=selected_features,
            n_clusters=None if self._radio_auto_k.isChecked() else self._spin_k.value(),
            max_k=10
        )

        self._start_clustering_worker(df, params, [current_variety], merge_all=False)

    def _start_clustering_worker(self, df, params, variety_groups, merge_all=False):
        """启动后台聚类工作线程"""
        # 创建进度对话框
        self._progress_dialog = QProgressDialog("正在计算聚类...", "取消", 0, 0, self)
        self._progress_dialog.setWindowTitle("计算中")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        # 创建工作线程（QObject + moveToThread 模式）
        self._worker_thread = QThread()
        self._worker = ClusteringWorker(df, params, variety_groups, merge_all)
        self._worker.moveToThread(self._worker_thread)

        # 连接信号
        self._worker.progress_signal.connect(self._on_worker_progress)
        self._worker.result_signal.connect(self._on_worker_result)
        self._worker.multi_result_signal.connect(self._on_worker_multi_result)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker.finished_signal.connect(self._on_worker_finished)
        self._worker_thread.started.connect(self._worker.run)
        self._progress_dialog.canceled.connect(self._worker.cancel)

        # 启动线程
        self._worker_thread.start()

        # 禁用按钮
        self._btn_run.setEnabled(False)
        self._btn_open.setEnabled(False)

    @Slot(str)
    def _on_worker_progress(self, message: str):
        """更新进度信息"""
        if self._progress_dialog:
            self._progress_dialog.setLabelText(message)

    @Slot(object)
    def _on_worker_result(self, result: ClusteringResult):
        """处理单品种聚类结果"""
        # 更新数据库
        self._data_manager.update_cluster_ids(result.df_with_clusters)

        # 刷新表格（保持当前品种过滤）
        self._table_model.refresh()

        # 显示可视化配置
        self._viz_group.setVisible(True)
        self._update_viz_comboboxes()

        # 更新状态栏（需求9.7）
        variety_name = self._combo_group.currentData()
        if variety_name:
            self._lbl_status.setText(
                f"{variety_name}聚类完成 | K={result.k} | 有效样本: {result.n_samples}"
            )

        # 自动绘图
        self._auto_plot(result)

        # 显示聚类中心信息
        self._show_centroids_info(result)

    @Slot(object)
    def _on_worker_multi_result(self, group_results: dict):
        """处理多品种聚类结果"""
        # 更新数据库
        self._data_manager.update_cluster_ids_by_group(group_results)

        # 刷新表格
        self._table_model.refresh()

        # 更新品种选择下拉框
        self._update_group_combobox(group_results)

        # 显示可视化配置
        self._viz_group.setVisible(True)
        self._update_viz_comboboxes()

        # 更新状态栏（需求9.7）
        total_groups = len(group_results)
        k_values = set(r.k for r in group_results.values())
        k_info = f"K={','.join(str(k) for k in sorted(k_values))}" if len(k_values) <= 3 else f"多K值"
        self._lbl_status.setText(
            f"多品种聚类完成 | {total_groups}个品种 | {k_info}"
        )

        # 自动绘图（显示第一个品种）
        if group_results:
            first_group = list(group_results.keys())[0]
            result = group_results[first_group]
            self._auto_plot(result)

        # 显示多品种聚类中心信息
        self._show_multi_centroids_info(group_results)

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

        # 清理线程
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
            self._worker = None

        # 恢复按钮状态（考虑当前品种模式）
        group_name = self._combo_group.currentData()
        if group_name is None:
            # 全部品种模式 → 保持禁用
            self._btn_run.setEnabled(False)
        else:
            checked_count = sum(1 for cb in self._feature_checkboxes if cb.isChecked())
            self._btn_run.setEnabled(checked_count >= 2)
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

    def _update_group_combobox(self, groups, sample_counts=None):
        """
        更新品种选择下拉框

        Args:
            groups: 品种名称列表 [str] 或 {品种名: ClusteringResult} 字典
            sample_counts: 可选，{品种名: 样本数} 字典
        """
        self._combo_group.blockSignals(True)
        self._combo_group.clear()

        # 添加"全部品种(仅查看)"选项（需求9.2.2）
        self._combo_group.addItem("全部品种(仅查看)", None)

        # 获取品种名称列表
        if isinstance(groups, dict):
            group_names = list(groups.keys())
        else:
            group_names = list(groups) if groups else []

        # 获取样本数
        if sample_counts is None:
            if isinstance(groups, dict):
                # 尝试从 DataManager 获取样本数
                try:
                    sample_counts = self._data_manager.get_variety_sample_counts()
                except:
                    sample_counts = {}
            else:
                sample_counts = {}

        # 添加各个品种（显示样本数，需求9.2.2）
        for name in sorted(group_names):
            count = sample_counts.get(name, 0)
            self._combo_group.addItem(f"{name}(样本数:{count})", name)

        self._combo_group.blockSignals(False)

    @Slot(int)
    def _on_group_changed(self, index: int):
        """品种选择变化处理（需求9.2.3 联动逻辑）"""
        group_name = self._combo_group.currentData()

        # 1. 更新表格过滤
        self._table_model.set_variety_filter(group_name)

        # 2. 更新聚类按钮状态（全部品种禁用聚类）
        if group_name is None:
            # 全部品种(仅查看)模式 → 禁用聚类
            self._btn_run.setEnabled(False)
            self._btn_run.setToolTip("请选择单一品种进行聚类")
        else:
            # 单品种模式 → 按特征选择状态启用
            checked_count = sum(1 for cb in self._feature_checkboxes if cb.isChecked())
            self._btn_run.setEnabled(checked_count >= 2)
            self._btn_run.setToolTip("")

        # 3. 更新状态栏的当前品种信息
        variety_counts = self._data_manager.get_variety_sample_counts()
        if variety_counts:
            total_varieties = len(variety_counts)
            summary = "、".join([
                f"{name}({count}条)"
                for name, count in sorted(variety_counts.items())
            ])
            current = group_name if group_name else "全部"
            status = f"已加载 {total_varieties} 个品种：{summary} | 当前：{current}"
            self._lbl_status.setText(status)

        # 4. 更新图表（需求9.5）
        features = self._get_selected_features()
        if len(features) < 2:
            return

        df = self._table_model._data  # 使用已过滤的数据
        if df is None or df.empty:
            return

        use_pca = self._chk_pca.isChecked()
        if group_name is None:
            # 全部品种模式 → 按品种名着色
            self._canvas.plot_clusters(
                df, features, use_pca=use_pca,
                color_by='variety_name'
            )
        else:
            # 单品种模式 → 按 cluster_id 着色
            self._canvas.plot_clusters(
                df, features, use_pca=use_pca,
                color_by='cluster_id', variety_name=group_name
            )

    def _auto_plot(self, result: ClusteringResult):
        """自动绘制聚类结果图（需求9.5）"""
        group_name = self._combo_group.currentData()

        # 使用表格模型当前已过滤的数据
        df = self._table_model._data
        if df is None or df.empty:
            return

        features = self._get_selected_features()
        if len(features) >= 2:
            use_pca = self._chk_pca.isChecked()
            if group_name is None:
                # 全部品种模式 → 按品种名着色
                self._canvas.plot_clusters(
                    df, features, use_pca=use_pca,
                    color_by='variety_name'
                )
            else:
                # 单品种模式 → 按 cluster_id 着色
                self._canvas.plot_clusters(
                    df, features, use_pca=use_pca,
                    color_by='cluster_id', variety_name=group_name,
                    n_clusters=result.k
                )

    def _on_plot_update(self):
        """更新图表按钮点击（需求9.5）"""
        use_pca = self._chk_pca.isChecked()
        all_features = self._get_selected_features()

        if use_pca and len(all_features) > 2:
            # PCA模式：使用所有选中特征进行降维
            features = all_features
        else:
            # 非PCA模式：使用X/Y下拉框指定的两个特征
            feature_x = self._combo_x.currentData()
            feature_y = self._combo_y.currentData()
            if not feature_x or not feature_y:
                return
            features = [feature_x, feature_y]
            use_pca = False  # 只有2个特征无需PCA

        group_name = self._combo_group.currentData()

        # 使用表格模型当前已过滤的数据
        df = self._table_model._data
        if df is None or df.empty:
            return

        if group_name is None:
            self._canvas.plot_clusters(
                df, features, use_pca=use_pca,
                color_by='variety_name'
            )
        else:
            self._canvas.plot_clusters(
                df, features, use_pca=use_pca,
                color_by='cluster_id', variety_name=group_name
            )

    def _show_centroids_info(self, result: ClusteringResult):
        """显示聚类中心信息"""
        centroids_df = result.centroids_df
        if centroids_df is None or centroids_df.empty:
            return

        msg = f"聚类完成！K = {result.k}，有效样本数 = {result.n_samples}\n\n"
        msg += "聚类中心特征值:\n\n"
        for idx, row in centroids_df.iterrows():
            msg += f"类别 {idx}:\n"
            for col in centroids_df.columns:
                msg += f"  {get_display_name(col)}: {row[col]:.4f}\n"
            msg += "\n"

        QMessageBox.information(self, "聚类中心", msg)

    def _show_multi_centroids_info(self, group_results: dict):
        """显示多品种聚类中心信息"""
        msg = f"多品种聚类完成！共处理 {len(group_results)} 个品种\n\n"

        for group_name, result in group_results.items():
            msg += f"=== {group_name} ===\n"
            msg += f"K = {result.k}，有效样本数 = {result.n_samples}\n"
            msg += "聚类中心特征值:\n"
            for idx, row in result.centroids_df.iterrows():
                msg += f"  类别 {idx}: "
                values = [f"{get_display_name(col)}={row[col]:.2f}" for col in result.centroids_df.columns]
                msg += ", ".join(values) + "\n"
            msg += "\n"

        QMessageBox.information(self, "多品种聚类中心", msg)

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
            use_chinese = self._chk_chinese_headers.isChecked()
            split_by_group = self._chk_split_by_group.isChecked()
            self._data_manager.export_to_excel(
                file_path,
                use_chinese_headers=use_chinese,
                split_by_group=split_by_group
            )
            QMessageBox.information(self, "成功", f"结果已导出到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self._worker and self._worker_thread:
            self._worker.cancel()
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._data_manager.close()
        event.accept()
