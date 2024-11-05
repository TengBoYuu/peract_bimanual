import plotly.graph_objects as go
import torch
import webbrowser
import os

# 初始化示例点
right = torch.tensor([[0.5, 0.5, 0.5]])  # right 点
right_gt = torch.tensor([[0.6, 0.6, 0.6]])  # right_gt 点

# 将 right 和 right_gt 点合并
points = torch.cat((right, right_gt), dim=0).cpu().numpy()

# 分别设置每个点的颜色和大小
colors = ['red', 'green']  # 为 right 和 right_gt 设置不同的颜色
sizes = [5, 10]  # right 的点小，right_gt 的点大

# 创建 Plotly 3D 散点图
fig = go.Figure()

# 添加 right 点
fig.add_trace(go.Scatter3d(
    x=[points[0, 0]],
    y=[points[0, 1]],
    z=[points[0, 2]],
    mode='markers',
    marker=dict(size=sizes[0], color=colors[0]),
    name='right'
))

# 添加 right_gt 点
fig.add_trace(go.Scatter3d(
    x=[points[1, 0]],
    y=[points[1, 1]],
    z=[points[1, 2]],
    mode='markers',
    marker=dict(size=sizes[1], color=colors[1]),
    name='right_gt'
))

# 设置标题和坐标轴标签
fig.update_layout(
    title="Point Cloud Visualization with Right and Right GT",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# 保存图表为 HTML 文件
output_file = "3d_scatter_plot.html"
fig.write_html(output_file)

# 自动在浏览器中打开生成的 HTML 文件
# 确保路径是绝对路径
file_path = os.path.abspath(output_file)
webbrowser.open(f"/mnt/disk_1/tengbo/peract_bimanual-real")

print(f"The 3D scatter plot has been saved to {output_file} and opened in your browser.")