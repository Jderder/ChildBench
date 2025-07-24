import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class JSONLFilterApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JSONL高级分析工具")
        self.root.geometry("800x600")

        self.filepath = ""
        self.analysis_result = {}
        self.conditions = {}

        self.create_widgets()

    def create_widgets(self):
        # 文件选择部分
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, fill=tk.X)

        tk.Button(
            file_frame,
            text="选择JSONL文件",
            command=self.select_file
        ).pack(side=tk.LEFT)

        self.file_label = tk.Label(file_frame, text="未选择文件", anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # 主分析区域
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧条件设置面板
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 字段选择部分
        condition_frame = tk.LabelFrame(left_panel, text="筛选条件设置")
        condition_frame.pack(fill=tk.X, pady=5)

        tk.Label(condition_frame, text="选择字段:").pack(anchor=tk.W)
        self.field_combobox = ttk.Combobox(condition_frame, state="readonly")
        self.field_combobox.pack(fill=tk.X, padx=5, pady=2)
        self.field_combobox.bind("<<ComboboxSelected>>", self.field_selected)

        tk.Label(condition_frame, text="选择值:").pack(anchor=tk.W)
        self.value_combobox = ttk.Combobox(condition_frame, state="readonly")
        self.value_combobox.pack(fill=tk.X, padx=5, pady=2)

        self.add_btn = tk.Button(
            condition_frame,
            text="添加条件",
            state=tk.DISABLED,
            command=self.add_condition
        )
        self.add_btn.pack(pady=5)

        # 已选条件显示
        self.conditions_text = tk.Text(
            left_panel,
            height=8,
            state=tk.DISABLED
        )
        self.conditions_text.pack(fill=tk.X, pady=5)

        # 操作按钮
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            btn_frame,
            text="执行筛选",
            command=self.apply_filter
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            btn_frame,
            text="清除条件",
            command=self.clear_conditions
        ).pack(side=tk.LEFT, padx=2)

        # 字段统计部分
        stats_frame = tk.LabelFrame(left_panel, text="字段分布统计")
        stats_frame.pack(fill=tk.X, pady=5)

        tk.Label(stats_frame, text="统计字段:").pack(anchor=tk.W)
        self.stats_combobox = ttk.Combobox(stats_frame, state="readonly")
        self.stats_combobox.pack(fill=tk.X, padx=5, pady=2)

        self.stats_btn = tk.Button(
            stats_frame,
            text="统计分布",
            state=tk.DISABLED,
            command=self.show_field_stats
        )
        self.stats_btn.pack(pady=5)

        # 右侧结果显示面板
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 结果显示区域
        self.result_text = tk.Text(
            right_panel,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 图表区域
        self.figure = plt.figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部按钮
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            bottom_frame,
            text="保存结果",
            command=self.save_results
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            bottom_frame,
            text="退出",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)

    def select_file(self):
        """选择JSONL文件并分析"""
        self.filepath = filedialog.askopenfilename(
            title="选择JSONL文件",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )

        if self.filepath:
            self.file_label.config(text=self.filepath)
            self.analyze_file()

    def analyze_file(self):
        """分析JSONL文件结构"""
        fields = set()
        field_values = defaultdict(set)
        total_lines = 0

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    for field, value in data.items():
                        fields.add(field)
                        field_values[field].add(str(value))
                except json.JSONDecodeError:
                    continue

        self.analysis_result = {
            'fields': sorted(fields),
            'field_values': {k: sorted(v) for k, v in field_values.items()},
            'total_lines': total_lines
        }

        # 更新字段下拉框
        self.field_combobox['values'] = self.analysis_result['fields']
        self.stats_combobox['values'] = self.analysis_result['fields']

        if self.analysis_result['fields']:
            self.field_combobox.current(0)
            self.field_selected()
            self.stats_btn.config(state=tk.NORMAL)

        messagebox.showinfo(
            "文件分析完成",
            f"共分析 {total_lines} 行数据\n发现字段: {', '.join(fields)}"
        )

    def field_selected(self, event=None):
        """字段选择事件"""
        selected_field = self.field_combobox.get()
        if selected_field in self.analysis_result['field_values']:
            values = self.analysis_result['field_values'][selected_field]
            self.value_combobox['values'] = values
            if values:
                self.value_combobox.current(0)
                self.add_btn.config(state=tk.NORMAL)

    def add_condition(self):
        """添加筛选条件"""
        field = self.field_combobox.get()
        value = self.value_combobox.get()

        if field and value:
            self.conditions[field] = value
            self.update_conditions_display()

    def update_conditions_display(self):
        """更新已选条件显示"""
        self.conditions_text.config(state=tk.NORMAL)
        self.conditions_text.delete(1.0, tk.END)

        for field, value in self.conditions.items():
            self.conditions_text.insert(tk.END, f"{field} = {value}\n")

        self.conditions_text.config(state=tk.DISABLED)

    def clear_conditions(self):
        """清除所有条件"""
        self.conditions = {}
        self.update_conditions_display()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.figure.clear()
        self.canvas.draw()

    def apply_filter(self):
        """执行筛选"""
        if not self.conditions:
            messagebox.showwarning("警告", "请至少添加一个筛选条件")
            return

        matched_lines = 0
        results = []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    match = True
                    for field, value in self.conditions.items():
                        if str(data.get(field, '')) != value:
                            match = False
                            break
                    if match:
                        matched_lines += 1
                        results.append(data)
                except json.JSONDecodeError:
                    continue

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END,
                                f"总行数: {self.analysis_result['total_lines']}\n"
                                f"匹配行数: {matched_lines}\n"
                                f"匹配率: {matched_lines / self.analysis_result['total_lines']:.1%}\n\n"
                                f"筛选条件: {self.conditions}\n\n"
                                "前10条匹配记录:\n"
                                )

        for i, item in enumerate(results[:10], 1):
            self.result_text.insert(tk.END, f"{i}. {json.dumps(item, ensure_ascii=False)}\n")

        self.result_text.config(state=tk.DISABLED)

    def show_field_stats(self):
        """显示字段分布统计"""
        if not self.filepath:
            return

        field = self.stats_combobox.get()
        if not field:
            messagebox.showwarning("警告", "请选择要统计的字段")
            return

        # 统计分布
        counter = defaultdict(int)
        total = 0

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # 检查是否满足筛选条件
                    match = True
                    for cond_field, cond_value in self.conditions.items():
                        if str(data.get(cond_field, '')) != cond_value:
                            match = False
                            break

                    if match:
                        value = str(data.get(field, '缺失'))
                        counter[value] += 1
                        total += 1
                except json.JSONDecodeError:
                    continue

        if not counter:
            messagebox.showinfo("信息", "没有匹配的数据")
            return

        # 显示统计结果
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END,
                                f"字段 '{field}' 的分布统计 (基于当前筛选条件)\n"
                                f"总匹配记录数: {total}\n\n"
                                )

        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for value, count in sorted_items:
            percent = count / total * 100
            self.result_text.insert(tk.END, f"{value}: {count} ({percent:.1f}%)\n")

        self.result_text.config(state=tk.DISABLED)

        # 绘制图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # 如果类别太多，只显示前20个
        if len(labels) > 20:
            labels = labels[:20] + ['其他']
            values = values[:20] + [sum(values[20:])]

        ax.bar(labels, values)
        ax.set_title(f"字段 '{field}' 的分布")
        ax.set_ylabel("数量")
        ax.tick_params(axis='x', rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()

    def save_results(self):
        """保存筛选结果"""
        if not self.filepath:
            return

        save_path = filedialog.asksaveasfilename(
            title="保存筛选结果",
            initialfile="filtered_results.jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )

        if save_path:
            if not save_path.endswith('.jsonl'):
                save_path += '.jsonl'

            matched_data = []
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        match = True
                        for field, value in self.conditions.items():
                            if str(data.get(field, '')) != value:
                                match = False
                                break
                        if match:
                            matched_data.append(data)
                    except json.JSONDecodeError:
                        continue

            with open(save_path, 'w', encoding='utf-8') as f:
                for item in matched_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            messagebox.showinfo("完成", f"结果已保存到: {save_path}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = JSONLFilterApp()
    app.run()