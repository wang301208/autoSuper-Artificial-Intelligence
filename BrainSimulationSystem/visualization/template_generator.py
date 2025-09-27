"""
模板生成器

为可视化服务器创建默认HTML模板。
"""

import os


def create_default_templates():
    """创建默认HTML模板"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    # 确保模板目录存在
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    # 创建基础模板
    create_base_template(template_dir)
    
    # 创建首页模板
    create_index_template(template_dir)
    
    # 创建可视化页面模板
    create_visualization_template(template_dir)
    
    # 创建神经网络页面模板
    create_network_template(template_dir)
    
    # 创建认知过程页面模板
    create_cognitive_template(template_dir)


def create_base_template(template_dir):
    """创建基础模板"""
    base_template = os.path.join(template_dir, 'base.html')
    if not os.path.exists(base_template):
        with open(base_template, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}大脑模拟系统{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <nav>
            <div class="logo">大脑模拟系统</div>
            <ul>
                <li><a href="/">首页</a></li>
                <li><a href="/visualization">可视化</a></li>
                <li><a href="/network">神经网络</a></li>
                <li><a href="/cognitive">认知过程</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        <p>&copy; 2025 大脑模拟系统</p>
    </footer>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>""")


def create_index_template(template_dir):
    """创建首页模板"""
    index_template = os.path.join(template_dir, 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}大脑模拟系统 - 首页{% endblock %}

{% block content %}
<div class="container">
    <h1>欢迎使用大脑模拟系统</h1>
    
    <div class="intro">
        <p>大脑模拟系统是一个用于模拟神经元网络、突触连接和认知过程的平台。</p>
        <p>通过本系统，您可以：</p>
        <ul>
            <li>观察神经元网络活动</li>
            <li>模拟认知过程</li>
            <li>研究神经调质对认知的影响</li>
            <li>可视化注意力和工作记忆过程</li>
        </ul>
    </div>
    
    <div class="quick-links">
        <h2>快速导航</h2>
        <div class="links-grid">
            <a href="/visualization" class="link-card">
                <h3>可视化面板</h3>
                <p>查看神经活动和认知过程的实时可视化</p>
            </a>
            <a href="/network" class="link-card">
                <h3>神经网络</h3>
                <p>探索神经元网络结构和突触连接</p>
            </a>
            <a href="/cognitive" class="link-card">
                <h3>认知过程</h3>
                <p>观察注意力、工作记忆和决策过程</p>
            </a>
        </div>
    </div>
    
    <div class="system-status">
        <h2>系统状态</h2>
        <div id="status-panel">
            <div class="status-item">
                <span class="label">模拟状态:</span>
                <span id="simulation-status" class="value">未运行</span>
            </div>
            <div class="status-item">
                <span class="label">认知状态:</span>
                <span id="cognitive-state" class="value">-</span>
            </div>
            <div class="status-item">
                <span class="label">神经元活动:</span>
                <span id="neuron-activity" class="value">-</span>
            </div>
        </div>
        
        <div class="controls">
            <button id="start-simulation" class="btn primary">开始模拟</button>
            <button id="stop-simulation" class="btn secondary" disabled>停止模拟</button>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    // 页面加载完成后执行
    document.addEventListener('DOMContentLoaded', function() {
        // 获取元素
        const startBtn = document.getElementById('start-simulation');
        const stopBtn = document.getElementById('stop-simulation');
        const simulationStatus = document.getElementById('simulation-status');
        const cognitiveState = document.getElementById('cognitive-state');
        const neuronActivity = document.getElementById('neuron-activity');
        
        // 定期更新状态
        function updateStatus() {
            fetch('/api/simulation/status')
                .then(response => response.json())
                .then(data => {
                    simulationStatus.textContent = data.running ? '运行中' : '未运行';
                    startBtn.disabled = data.running;
                    stopBtn.disabled = !data.running;
                })
                .catch(error => console.error('获取模拟状态失败:', error));
                
            fetch('/api/cognitive/state')
                .then(response => response.json())
                .then(data => {
                    cognitiveState.textContent = data.cognitive_state || '-';
                })
                .catch(error => console.error('获取认知状态失败:', error));
        }
        
        // 初始更新状态
        updateStatus();
        
        // 每2秒更新一次状态
        setInterval(updateStatus, 2000);
        
        // 绑定按钮事件
        startBtn.addEventListener('click', function() {
            fetch('/api/simulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ steps: 100, interval: 0.5 })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    simulationStatus.textContent = '运行中';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                }
            })
            .catch(error => console.error('启动模拟失败:', error));
        });
        
        stopBtn.addEventListener('click', function() {
            fetch('/api/simulation/stop', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'stopped') {
                    simulationStatus.textContent = '未运行';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            })
            .catch(error => console.error('停止模拟失败:', error));
        });
    });
</script>
{% endblock %}""")


def create_visualization_template(template_dir):
    """创建可视化页面模板"""
    visualization_template = os.path.join(template_dir, 'visualization.html')
    if not os.path.exists(visualization_template):
        with open(visualization_template, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}大脑模拟系统 - 可视化{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/visualization.css') }}">
{% endblock %}

{% block content %}
<div class="container">
    <h1>神经活动可视化</h1>
    
    <div class="visualization-controls">
        <div class="control-group">
            <button id="start-simulation" class="btn primary">开始模拟</button>
            <button id="stop-simulation" class="btn secondary" disabled>停止模拟</button>
        </div>
        <div class="control-group">
            <label for="update-interval">更新间隔 (ms):</label>
            <input type="range" id="update-interval" min="100" max="2000" step="100" value="500">
            <span id="interval-value">500</span>
        </div>
    </div>
    
    <div class="visualization-grid">
        <div class="visualization-panel">
            <h2>神经元活动</h2>
            <div class="chart-container">
                <canvas id="neural-activity-chart"></canvas>
            </div>
        </div>
        
        <div class="visualization-panel">
            <h2>神经调质水平</h2>
            <div class="chart-container">
                <canvas id="neuromodulators-chart"></canvas>
            </div>
        </div>
        
        <div class="visualization-panel">
            <h2>注意力焦点</h2>
            <div class="chart-container">
                <canvas id="attention-chart"></canvas>
            </div>
        </div>
        
        <div class="visualization-panel">
            <h2>工作记忆内容</h2>
            <div id="memory-content" class="memory-container">
                <div class="memory-placeholder">未加载数据</div>
            </div>
        </div>
    </div>
    
    <div class="status-panel">
        <h2>系统状态</h2>
        <div class="status-grid">
            <div class="status-item">
                <span class="label">模拟状态:</span>
                <span id="simulation-status" class="value">未运行</span>
            </div>
            <div class="status-item">
                <span class="label">当前步骤:</span>
                <span id="current-step" class="value">0</span>
            </div>
            <div class="status-item">
                <span class="label">认知状态:</span>
                <span id="cognitive-state" class="value">-</span>
            </div>
            <div class="status-item">
                <span class="label">进度:</span>
                <div class="progress-bar">
                    <div id="progress" class="progress" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="{{ url_for('static', filename='js/visualization.js') }}"></script>
{% endblock %}""")


def create_network_template(template_dir):
    """创建神经网络页面模板"""
    network_template = os.path.join(template_dir, 'network.html')
    if not os.path.exists(network_template):
        with open(network_template, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}大脑模拟系统 - 神经网络{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/network.css') }}">
{% endblock %}

{% block content %}
<div class="container">
    <h1>神经网络结构</h1>
    
    <div class="network-controls">
        <div class="control-group">
            <label for="network-view">视图:</label>
            <select id="network-view">
                <option value="layers">分层视图</option>
                <option value="connections">连接视图</option>
                <option value="activity">活动视图</option>
            </select>
        </div>
        <div class="control-group">
            <label for="zoom-level">缩放:</label>
            <input type="range" id="zoom-level" min="50" max="200" step="10" value="100">
            <span id="zoom-value">100%</span>
        </div>
    </div>
    
    <div class="network-container">
        <div id="network-visualization"></div>
    </div>
    
    <div class="network-details">
        <h2>网络详情</h2>
        <div class="details-grid">
            <div class="detail-item">
                <span class="label">神经元数量:</span>
                <span id="neuron-count" class="value">-</span>
            </div>
            <div class="detail-item">
                <span class="label">突触数量:</span>
                <span id="synapse-count" class="value">-</span>
            </div>
            <div class="detail-item">
                <span class="label">活跃神经元:</span>
                <span id="active-neurons" class="value">-</span>
            </div>
            <div class="detail-item">
                <span class="label">平均活动水平:</span>
                <span id="avg-activity" class="value">-</span>
            </div>
        </div>
    </div>
    
    <div class="selected-element-info">
        <h2>选中元素信息</h2>
        <div id="element-info" class="info-container">
            <div class="info-placeholder">未选中任何元素</div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="{{ url_for('static', filename='js/network.js') }}"></script>
{% endblock %}""")


def create_cognitive_template(template_dir):
    """创建认知过程页面模板"""
    cognitive_template = os.path.join(template_dir, 'cognitive.html')
    if not os.path.exists(cognitive_template):
        with open(cognitive_template, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}大脑模拟系统 - 认知过程{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/cognitive.css') }}">
{% endblock %}

{% block content %}
<div class="container">
    <h1>认知过程可视化</h1>
    
    <div class="cognitive-controls">
        <div class="control-group">
            <button id="start-simulation" class="btn primary">开始模拟</button>
            <button id="stop-simulation" class="btn secondary" disabled>停止模拟</button>
        </div>
        <div class="control-group">
            <label for="cognitive-view">视图:</label>
            <select id="cognitive-view">
                <option value="attention">注意力过程</option>
                <option value="memory">工作记忆</option>
                <option value="decision">决策过程</option>
                <option value="integrated">综合视图</option>
            </select>
        </div>
    </div>
    
    <div class="cognitive-grid">
        <div class="cognitive-panel">
            <h2>认知状态变化</h2>
            <div class="chart-container">
                <canvas id="cognitive-state-chart"></canvas>
            </div>
        </div>
        
        <div class="cognitive-panel">
            <h2>神经调质影响</h2>
            <div class="chart-container">
                <canvas id="neuromodulator-effect-chart"></canvas>
            </div>
        </div>
        
        <div class="cognitive-panel">
            <h2>注意力-工作记忆交互</h2>
            <div id="attention-memory-interaction" class="interaction-container"></div>
        </div>
        
        <div class="cognitive-panel">
            <h2>认知过程流</h2>
            <div id="cognitive-flow" class="flow-container"></div>
        </div>
    </div>
    
    <div class="cognitive-details">
        <h2>认知详情</h2>
        <div id="cognitive-details-content" class="details-content">
            <div class="details-placeholder">未加载数据</div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="{{ url_for('static', filename='js/cognitive.js') }}"></script>
{% endblock %}""")