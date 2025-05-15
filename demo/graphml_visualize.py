import networkx as nx  # 用于读取和处理 graphml 图结构
import json            # 用于处理 JSON 数据
import os              # 用于文件和路径操作
import webbrowser      # 用于自动打开浏览器
import http.server     # 用于启动本地 HTTP 服务
import socketserver    # 用于创建 TCP 服务器
import threading       # 用于多线程（后台启动服务器）

# 将 GraphML 文件读取为 networkx 图对象，并转换为 D3.js 可用的 JSON 格式
def graphml_to_json(graphml_file):
    G = nx.read_graphml(graphml_file)      # 读取 graphml 文件为 networkx 图
    data = nx.node_link_data(G)            # 转换为 node-link 格式（适合 D3.js）
    return json.dumps(data)                # 序列化为 JSON 字符串

# 生成 HTML 文件，内嵌 D3.js 脚本用于可视化
def create_html(html_path):
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        svg {
            width: 100%;
            height: 100%;
        }
        .links line {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .nodes circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node-label {
            font-size: 12px;
            pointer-events: none;
        }
        .link-label {
            font-size: 10px;
            fill: #666;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .link:hover .link-label {
            opacity: 1;
        }
        .tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: lightsteelblue;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 300px;
        }
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        .legend-item {
            margin: 5px 0;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <svg></svg>
    <div class="tooltip"></div>
    <div class="legend"></div>
    <script type="text/javascript" src="./graph_json.js"></script>
    <script>
        // 读取 graph_json.js 里的全局变量
        const graphData = graphJson;
        
        const svg = d3.select("svg"),
            width = window.innerWidth,
            height = window.innerHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        // 获取所有实体类型，用于着色
        const entityTypes = [...new Set(graphData.nodes.map(d => d.entity_type))];
        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

        // 配置 D3 力导向布局
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(30));

        // 绘制边
        const linkGroup = g.append("g")
            .attr("class", "links")
            .selectAll("g")
            .data(graphData.links)
            .enter().append("g")
            .attr("class", "link");

        const link = linkGroup.append("line")
            .attr("stroke-width", d => Math.sqrt(d.value));

        // 边的标签
        const linkLabel = linkGroup.append("text")
            .attr("class", "link-label")
            .text(d => d.description || "");

        // 绘制节点
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", 5)
            .attr("fill", d => color(d.entity_type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // 节点标签
        const nodeLabel = g.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.id);

        // 鼠标悬停提示框
        const tooltip = d3.select(".tooltip");

        node.on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`<strong>${d.id}</strong><br>Entity Type: ${d.entity_type}<br>Description: ${d.description || "N/A"}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        // 图例
        const legend = d3.select(".legend");
        entityTypes.forEach(type => {
            legend.append("div")
                .attr("class", "legend-item")
                .html(`<span class="legend-color" style="background-color: ${color(type)}"></span>${type}`);
        });

        // tick 回调，实时更新节点和边的位置
        simulation
            .nodes(graphData.nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(graphData.links);

        function ticked() {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle");

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            nodeLabel
                .attr("x", d => d.x + 8)
                .attr("y", d => d.y + 3);
        }

        // 拖拽事件
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // 缩放支持
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", zoomed);

        svg.call(zoom);

        function zoomed(event) {
            g.attr("transform", event.transform);
        }

    </script>
</body>
</html>
    '''
    # 写入 HTML 文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# 生成 graph_json.js 文件，内容为 var graphJson = {...}
def create_json(json_data, json_path):
    # 处理字符串，避免转义问题
    json_data = "var graphJson = " + json_data.replace('\\"', '').replace("'", "\\'").replace("\n", "")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_data)

# 启动本地 HTTP 服务器，便于浏览器访问 HTML 文件
def start_server(port):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Server started at http://localhost:{port}")
        httpd.serve_forever()

# 主流程：生成 json、html，启动服务器并自动打开浏览器
def visualize_graphml(graphml_file, html_path, port=8000):
    json_data = graphml_to_json(graphml_file)  # 读取 graphml 并转为 json
    html_dir = os.path.dirname(html_path)
    if html_dir and not os.path.exists(html_dir):  # 只在目录非空时创建
        os.makedirs(html_dir)
    json_path = os.path.join(html_dir, 'graph_json.js')
    create_json(json_data, json_path)          # 写入 json 文件
    create_html(html_path)                     # 写入 html 文件
    # 启动 HTTP 服务器（后台线程）
    server_thread = threading.Thread(target=start_server, args=(port,))
    server_thread.daemon = True
    server_thread.start()
    
    # 自动打开浏览器
    webbrowser.open(f'http://localhost:{port}/{html_path}')
    
    print("Visualization is ready. Press Ctrl+C to exit.")
    try:
        # 主线程保持运行，直到 Ctrl+C
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")

# 用法示例
if __name__ == "__main__":
    graphml_file = r"/Users/huawuque/Desktop/RAG实验/nano-graphrag/demo/dickens/graph_chunk_entity_relation.graphml"  # 替换为你的 GraphML 路径
    html_path = "graph_visualization.html"
    visualize_graphml(graphml_file, html_path, 11236)