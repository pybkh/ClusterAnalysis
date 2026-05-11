完善计划：多品种识别与选择功能（优化.md 第9章）                                                                                                                        
   
     背景                                                                                                                                                                   
                                                            
     基于"优化.md"第9章需求文档，对植物表型聚类分析系统进行多品种功能完善。当前代码已初步支持多品种，但存在品种提取逻辑不匹配、数据流bug、UI联动不完整等问题。
                                                             
     ---                                  
     阶段1：修复核心Bug（最高优先级）

     1.1 修复 get_data_by_group — models/models.py

     - 问题：get_data_for_clustering 返回的 DataFrame 缺少 variety_name 列，导致 Worker 无法按品种筛选数据
     - 修复：在 get_data_by_group 的 SELECT 查询中始终包含 variety_name 列
     - 文件：models/models.py 第508行，cols 添加 'variety_name'

     1.2 重写 _extract_variety_name — models/utils.py

     - 问题：当前用的是"行模式"（同一行找相邻列），需求要求"列模式"（找到"品种名称"单元格后，取下一行同列的值）
     - 重写策略优先级：
       a. 内联格式："品种名称: 水稻" 或 "品种名称：水稻"
       b. 列模式（新增）：找到含"品种名称"的单元格，取下一行同列的值。对应需求9.1.2示例（第1行D列标题=品种名称，第2行D列=水稻）
       c. 同行邻列：同一行其他列的值（兼容旧格式）
     - 新增 _extract_column_value() 辅助函数

     ---
     阶段2：品种名称提取增强

     2.1 品种名称标准化 — models/utils.py

     - 新增 normalize_variety_name(name, seen_names, index) 函数
     - 去除首尾空格、换行符
     - 空值或纯数字 → "未知品种_N"（N为序号）

     2.2 Sheet名称后备 — models/models.py

     - 在 _process_data_blocks 中，当 variety_name 为默认值或空时，降级使用 source_sheet 名
     - 跟踪 seen_names set 避免重复

     2.3 手动输入后备 — views/views.py + models/models.py

     - 新增 DataManager.rename_variety(old_name, new_name) 方法
     - 在 _on_data_loaded 中检测"未知品种"时弹窗让用户输入

     ---
     阶段3：UI品种选择控件增强

     3.1 品种下拉框显示样本数 — views/views.py

     - 新增 DataManager.get_variety_sample_counts() 返回 {品种名: 样本数}
     - _update_group_combobox 改格式："全部品种(仅查看)" + "水稻(样本数:11)"

     3.2 品种切换联动逻辑 — views/views.py

     - 表格过滤：新增 DataManager.get_filtered_data(variety_name)，SqlTableModel 增加过滤方法
     - 切换品种时：表格同步过滤 + 图表同步更新
     - "全部品种"模式：显示所有行，品种列高亮

     3.3 "全部品种"禁用聚类 — views/views.py

     - _on_group_changed 中动态控制 _btn_run 状态
     - 全部品种选中 → 按钮禁用 + tooltip "请选择单一品种进行聚类"
     - 单品种选中 → 按特征复选框状态启用

     ---
     阶段4：聚类执行模式

     4.1 单品种聚类（默认）

     - _on_run_clustering 改为仅获取当前选中品种的数据进行聚类（get_data_by_group(features, variety_name)）
     - Worker 传入单元素 variety_group → 走 _run_single_group 路径

     4.2 全品种聚类（可选）

     - 左侧面板添加复选框 [☐ 合并所有品种聚类（不推荐）]
     - 勾选并点击运行时弹窗警告，确认后合并聚类
     - Worker 新增 _run_merged_group() 方法

     4.3 cluster_id 格式化显示

     - "全部品种"查看模式 → SqlTableModel.data() 中显示 "品种名_簇号" 格式
     - 数据库仍存原始整数，格式化在视图层处理

     ---
     阶段5：可视化品种区分

     5.1 按品种名着色 — views/views.py (ScatterCanvas)

     - plot_clusters 新增 color_by 参数：'cluster_id' 或 'variety_name'
     - 全部品种模式 → color_by='variety_name'，使用 Set1 配色
     - 单品种模式 → color_by='cluster_id'，使用 Set2 配色（现有逻辑）

     5.2 动态图表标题

     - 标题格式："水稻 - PCA聚类结果 (K=3)" 或 "品种分布图（2个品种）"

     ---
     阶段6：状态栏增强

     6.1 状态栏品种信息 — views/views.py

     - 加载完成时："已加载 2 个品种：水稻(11条)、油菜(6条) | 当前：水稻"
     - 聚类完成时："水稻聚类完成 | K=3 | 有效样本: 42"

     ---
     验证方案

     1. 单元测试：运行 python -c "from models import DataManager; from models.utils import normalize_variety_name; print('OK')"
     2. 功能验证：运行 python main.py，测试：
       - 加载多品种 Excel 文件（品种名称自动提取正确性）
       - 品种下拉框显示样本数
       - 切换品种 → 表格过滤 + 图表同步
       - 全部品种模式 → 图表按品种着色 + 聚类禁用
       - 单品种聚类 → 仅该品种数据参与
       - 全品种聚类复选框 → 弹窗警告 → 合并聚类
     3. 边缘情况：
       - 品种名称为空/纯数字的文件
       - 无品种名称的文件（降级使用 sheet 名）
       - 单品种文件（向后兼容）

     文件变更清单

     ┌────────────────────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
     │          文件          │ 修改类型  │                                               关键内容                                               │
     ├────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ models/utils.py        │ 重写+新增 │ _extract_variety_name 列模式优先；新增 normalize_variety_name                                        │
     ├────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ models/models.py       │ 修复+新增 │ get_data_by_group 加 variety_name；新增 get_filtered_data、get_variety_sample_counts、rename_variety │
     ├────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ views/views.py         │ 大量修改  │ 品种控件/切换联动/聚类模式/图表着色/状态栏                                                           │
     ├────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ controllers/workers.py │ 新增      │ _run_merged_group 合并聚类方法                                                                       │
     └────────────────────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘