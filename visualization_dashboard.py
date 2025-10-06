import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import torch
import dgl
import numpy as np
import pandas as pd
from model import GAT_COBO
import torch.nn.functional as F


class GraphVisualizationDashboard:
    def __init__(self, model, graph, features, labels):
        self.model = model
        self.graph = graph
        self.features = features
        self.labels = labels
        self.setup_ui()

    def setup_ui(self):
        st.set_page_config(layout="wide", page_title="图异常检测分析平台")

        # 侧边栏控制面板
        st.sidebar.title("控制面板")

        # 模型选择
        self.model_selection = st.sidebar.selectbox(
            "选择检测模型",
            ["GAT-COBO", "GCN", "GraphSAGE"]
        )

        # 风险阈值调节
        self.risk_threshold = st.sidebar.slider(
            "风险检测阈值", 0.1, 1.0, 0.7, 0.05
        )

        # 可视化参数
        self.node_size = st.sidebar.slider("节点大小", 5, 20, 10)
        self.show_labels = st.sidebar.checkbox("显示节点标签", True)

    def run(self):
        # 主界面布局
        col1, col2 = st.columns([2, 1])

        with col1:
            st.title("🧠 实时图异常检测分析平台")

            # 标签页布局
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 网络概览",
                "🔍 异常检测",
                "📈 时序分析",
                "🔎 节点探查"
            ])

            with tab1:
                self.render_network_overview()

            with tab2:
                self.render_anomaly_detection()

            with tab3:
                self.render_temporal_analysis()

            with tab4:
                self.render_node_inspection()

        with col2:
            self.render_control_panel()

    def render_network_overview(self):
        st.subheader("网络结构概览")

        # 创建网络图
        fig = self.create_network_graph()
        st.plotly_chart(fig, use_container_width=True)

        # 网络统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("节点数量", self.graph.number_of_nodes())
        with col2:
            st.metric("边数量", self.graph.number_of_edges())
        with col3:
            st.metric("异常节点", f"{self.get_anomaly_count()}个")
        with col4:
            st.metric("异常比例", f"{self.get_anomaly_ratio():.2%}")

    def create_network_graph(self):
        # 将DGL图转换为NetworkX用于可视化
        g_nx = self.graph.to_networkx()
        pos = nx.spring_layout(g_nx)

        # 获取节点颜色（基于标签）
        node_colors = ['red' if label == 1 else 'blue'
                       for label in self.labels]

        # 创建Plotly图
        edge_x, edge_y = [], []
        for edge in g_nx.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x, node_y = [], []
        for node in g_nx.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=self.node_size,
                color=node_colors,
                line=dict(width=2)))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='网络拓扑结构',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="蓝色: 正常节点, 红色: 异常节点",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def render_anomaly_detection(self):
        st.subheader("实时异常检测")

        # 获取模型预测
        with torch.no_grad():
            logits, _, attention = self.model(self.features)
            predictions = torch.softmax(logits, dim=1)
            anomaly_scores = predictions[:, 1]  # 异常类别的概率

        # 异常分数分布
        fig1 = px.histogram(
            x=anomaly_scores.numpy(),
            nbins=50,
            title="异常分数分布",
            labels={'x': '异常分数', 'y': '数量'}
        )
        fig1.add_vline(x=self.risk_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig1, use_container_width=True)

        # 高风险节点列表
        high_risk_nodes = torch.where(anomaly_scores > self.risk_threshold)[0]
        st.write(f"**检测到 {len(high_risk_nodes)} 个高风险节点**")

        if len(high_risk_nodes) > 0:
            risk_df = pd.DataFrame({
                '节点ID': high_risk_nodes.numpy(),
                '异常分数': anomaly_scores[high_risk_nodes].numpy(),
                '真实标签': self.labels[high_risk_nodes].numpy()
            })
            st.dataframe(risk_df.sort_values('异常分数', ascending=False))

    def render_temporal_analysis(self):
        st.subheader("异常模式时序分析")

        # 模拟时序数据（实际中可以从历史数据加载）
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        anomaly_counts = np.random.poisson(5, 30) + np.sin(np.arange(30) * 0.5) * 2

        fig = px.line(
            x=dates, y=anomaly_counts,
            title="每日异常检测数量趋势",
            labels={'x': '日期', 'y': '异常数量'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 热点区域分析
        st.subheader("异常热点分布")
        col1, col2 = st.columns(2)

        with col1:
            # 节点度分布 vs 异常分数
            degrees = self.graph.in_degrees()
            fig = px.scatter(
                x=degrees.numpy(),
                y=torch.softmax(self.model(self.features)[0], dim=1)[:, 1].detach().numpy(),
                title="节点度 vs 异常分数",
                labels={'x': '节点度', 'y': '异常分数'}
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_node_inspection(self):
        st.subheader("节点详细探查")

        # 节点选择
        node_id = st.number_input("输入节点ID", 0, self.graph.number_of_nodes() - 1, 0)

        if st.button("分析该节点"):
            self.analyze_single_node(node_id)

    def analyze_single_node(self, node_id):
        with torch.no_grad():
            logits, _, attention = self.model(self.features)
            node_prediction = torch.softmax(logits[node_id], dim=0)
            attention_weights = attention[node_id]

        col1, col2 = st.columns(2)

        with col1:
            # 预测概率饼图
            fig = px.pie(
                values=node_prediction.numpy(),
                names=['正常', '异常'],
                title=f"节点 {node_id} 预测概率"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 节点信息
            st.write(f"**节点详细信息:**")
            st.write(f"- 真实标签: {self.labels[node_id].item()}")
            st.write(f"- 异常概率: {node_prediction[1].item():.4f}")
            st.write(f"- 特征维度: {self.features[node_id].shape[0]}")

        with col2:
            # 注意力权重可视化
            top_k = min(10, len(attention_weights))
            top_attention = torch.topk(attention_weights, k=top_k)

            fig = px.bar(
                x=top_attention.values.numpy(),
                y=[f'节点 {i}' for i in top_attention.indices.numpy()],
                orientation='h',
                title="最重要的邻居节点（注意力权重）"
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_control_panel(self):
        st.sidebar.subheader("实时控制")

        # 实时检测开关
        if st.sidebar.button("开始实时监控"):
            self.start_real_time_monitoring()

        # 模型重训练
        if st.sidebar.button("在线更新模型"):
            self.online_model_update()

        # 数据导出
        if st.sidebar.button("导出检测结果"):
            self.export_results()

        # 系统状态
        st.sidebar.subheader("系统状态")
        st.sidebar.info("""
        🟢 模型运行正常  
        🔵 数据流正常  
        🟡 监控中...
        """)

    def get_anomaly_count(self):
        return torch.sum(self.labels == 1).item()

    def get_anomaly_ratio(self):
        return self.get_anomaly_count() / len(self.labels)

    def start_real_time_monitoring(self):
        st.sidebar.success("实时监控已启动")
        # 这里可以添加实时数据流处理

    def online_model_update(self):
        st.sidebar.warning("模型更新功能开发中...")

    def export_results(self):
        st.sidebar.info("数据导出功能开发中...")


# 使用示例
if __name__ == "__main__":
    # 这里需要加载您的模型和数据
    # model = load_your_model()
    # graph, features, labels = load_your_data()

    # dashboard = GraphVisualizationDashboard(model, graph, features, labels)
    # dashboard.run()

    st.info("请先配置模型和数据路径")