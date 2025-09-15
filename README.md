# AutoGPT v0.5.1 - 自主AI智能体平台

[![Version](https://img.shields.io/badge/version-0.5.1-blue.svg)](https://github.com/Significant-Gravitas/AutoGPT)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

AutoGPT是一个实验性的开源应用程序，展示了现代大型语言模型的能力。该程序由GPT-4驱动，将LLM"思维"链接在一起，自主实现您设定的任何目标。作为GPT-4完全自主运行的首批示例之一，AutoGPT推动了AI可能性的边界。

## 🏗️ 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "前端层"
        A[Flutter Web App] --> B[Agent Protocol Client]
        C[CLI Interface] --> B
    end
    
    subgraph "API网关层"
        B --> D[Agent Protocol Server]
        D --> E[FastAPI Router]
    end
    
    subgraph "核心智能体层"
        E --> F[Agent Manager]
        F --> G[Multi-Layer Agent]
        
        subgraph "智能体层级"
            G --> H[Governance Agent]
            H --> I[Capability Agent]
            I --> J[Execution Agent]
            J --> K[Evolution Agent]
        end
    end
    
    subgraph "核心服务层"
        G --> L[Planning System]
        G --> M[Memory System]
        G --> N[Knowledge Graph]
        G --> O[Learning System]
        
        L --> P[Prompt Strategies]
        M --> Q[Vector Storage]
        N --> R[Ontology Reasoner]
        O --> S[Experience Learner]
    end
    
    subgraph "能力层"
        G --> T[Ability Registry]
        T --> U[Built-in Abilities]
        T --> V[Plugin Abilities]
        
        U --> W[File Operations]
        U --> X[Web Browsing]
        U --> Y[Code Execution]
        U --> Z[System Commands]
    end
    
    subgraph "资源层"
        G --> AA[Resource Providers]
        AA --> BB[OpenAI Provider]
        AA --> CC[Memory Providers]
        AA --> DD[Storage Backends]
        
        BB --> EE[GPT-4/3.5 Models]
        CC --> FF[ChromaDB/Pinecone]
        DD --> GG[Local/GCS/S3]
    end
    
    subgraph "数据层"
        F --> HH[Agent Database]
        M --> II[Vector Database]
        G --> JJ[File Storage]
        
        HH --> KK[SQLite/PostgreSQL]
        II --> LL[ChromaDB/Redis]
        JJ --> MM[Local/Cloud Storage]
    end
```

### 核心组件说明

#### 1. 多层智能体架构
- **Governance Agent**: 顶层治理智能体，执行策略控制和安全检查
- **Capability Agent**: 能力选择智能体，负责选择和委托执行任务能力
- **Execution Agent**: 执行智能体，从执行计划中执行具体能力
- **Evolution Agent**: 进化智能体，基于过往结果适应任务和能力

#### 2. 核心系统
- **Planning System**: 规划系统，将用户意图转换为语言模型提示
- **Memory System**: 记忆系统，支持短期和长期记忆管理
- **Knowledge Graph**: 知识图谱，提供结构化知识存储和推理
- **Learning System**: 学习系统，从经验中学习和改进

## 🚀 新增功能特性

### v0.5.1 主要更新

#### 1. 架构重构
- ✨ **Monorepo架构**: 统一的工作空间管理多个子项目
- ✨ **分层智能体**: 实现多层智能体架构，提高决策质量
- ✨ **模块化设计**: 清晰的模块边界和依赖管理
- ✨ **配置系统重构**: 可序列化的智能体配置

#### 2. 增强的AI能力
- ✨ **多模态输入**: 支持文本+图像的多模态输入处理
- ✨ **知识图谱**: 集成本体推理和知识冲突检测
- ✨ **经验学习**: 从历史执行中学习和优化
- ✨ **量子计算支持**: 集成Qiskit进行量子算法研究

#### 3. 改进的用户体验
- ✨ **Flutter跨平台客户端**: 支持Web、移动端和桌面端
- ✨ **实时协作**: 多用户协作功能
- ✨ **可视化界面**: 任务管理和执行监控界面
- ✨ **响应式设计**: 适配不同屏幕尺寸

#### 4. 企业级特性
- ✨ **Agent Protocol**: 标准化的智能体通信协议
- ✨ **多存储后端**: 支持本地、GCS、S3存储
- ✨ **安全增强**: 命令执行安全控制和审计
- ✨ **监控告警**: 集成Sentry错误监控

## 📋 系统要求

### 基础要求
- **Python**: 3.10 或更高版本
- **Node.js**: 16+ (用于前端开发)
- **Flutter**: 3.x (用于移动端开发)
- **Docker**: 20.10+ (可选，用于容器化部署)

### 硬件要求
- **内存**: 最低 4GB RAM，推荐 8GB+
- **存储**: 最低 10GB 可用空间
- **网络**: 稳定的互联网连接（用于API调用）

## 🛠️ 安装部署

### 方式一：Docker部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd AutoGPT

# 2. 配置环境变量
cp backend/autogpt/.env.example backend/autogpt/.env
# 编辑 .env 文件，设置 OPENAI_API_KEY

# 3. 使用Docker Compose启动
cd backend/autogpt
docker-compose up --build
```

### 方式二：本地开发部署

```bash
# 1. 克隆项目
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd AutoGPT

# 2. 安装Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. 安装后端依赖
cd backend/autogpt
poetry install

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置必要的配置

# 5. 启动后端服务
poetry run autogpt serve

# 6. 安装前端依赖（新终端）
cd ../../frontend
flutter pub get

# 7. 启动前端应用
flutter run -d chrome --web-port 5000
```

### 方式三：生产环境部署

```bash
# 1. 构建生产镜像
docker build -t autogpt:latest --build-arg BUILD_TYPE=release .

# 2. 运行生产容器
docker run -d \
  --name autogpt-prod \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  -e DATABASE_STRING=postgresql://user:pass@host:5432/db \
  -v /path/to/data:/app/data \
  autogpt:latest serve
```

## 🔧 配置说明

### 环境变量配置

```bash
# OpenAI配置
OPENAI_API_KEY=your-openai-api-key
SMART_LLM=gpt-4-turbo
FAST_LLM=gpt-3.5-turbo

# 数据库配置
DATABASE_STRING=sqlite:///agent.db
# 或使用PostgreSQL: postgresql://user:pass@host:5432/dbname

# 存储配置
FILE_STORAGE_BACKEND=local  # local, gcs, s3
STORAGE_BUCKET=autogpt-bucket

# 服务器配置
AP_SERVER_PORT=8000
AP_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000

# 安全配置
EXECUTE_LOCAL_COMMANDS=false
RESTRICT_TO_WORKSPACE=true
SHELL_COMMAND_CONTROL=denylist
```

### 高级配置

```yaml
# config/prompt_settings.yaml
constraints:
  - 'Exclusively use the commands listed below.'
  - 'You can only act proactively, and are unable to start background jobs.'

resources:
  - 'Internet access for searches and information gathering.'
  - 'The ability to read and write files.'

best_practices:
  - 'Continuously review and analyze your actions.'
  - 'Every command has a cost, so be smart and efficient.'
```

## 📚 API接口规范

### Agent Protocol API

AutoGPT实现了标准的Agent Protocol规范，提供RESTful API接口：

#### 任务管理

```http
# 创建任务
POST /ap/v1/agent/tasks
Content-Type: application/json

{
  "input": "分析这个数据文件并生成报告",
  "additional_input": {}
}

# 获取任务列表
GET /ap/v1/agent/tasks

# 获取任务详情
GET /ap/v1/agent/tasks/{task_id}
```

#### 步骤执行

```http
# 执行步骤
POST /ap/v1/agent/tasks/{task_id}/steps
Content-Type: application/json

{
  "input": "继续执行下一步",
  "additional_input": {}
}

# 获取步骤列表
GET /ap/v1/agent/tasks/{task_id}/steps

# 获取步骤详情
GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}
```

#### 文件管理

```http
# 上传文件
POST /ap/v1/agent/tasks/{task_id}/artifacts
Content-Type: multipart/form-data

# 下载文件
GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}

# 获取文件列表
GET /ap/v1/agent/tasks/{task_id}/artifacts
```

### 响应格式

```json
{
  "task_id": "uuid-string",
  "input": "用户输入",
  "status": "created|running|completed|failed",
  "created_at": "2024-01-01T00:00:00Z",
  "modified_at": "2024-01-01T00:00:00Z",
  "artifacts": []
}
```

## 🧪 测试框架

### 运行测试

```bash
# 运行所有测试
poetry run pytest

# 运行单元测试
poetry run pytest tests/unit

# 运行集成测试
poetry run pytest tests/integration

# 生成覆盖率报告
poetry run pytest --cov=autogpt --cov-report=html
```

### 测试结构

```
tests/
├── unit/                 # 单元测试
│   ├── test_agents.py
│   ├── test_commands.py
│   └── test_config.py
├── integration/          # 集成测试
│   ├── test_agent_protocol.py
│   └── test_file_operations.py
└── conftest.py          # 测试配置
```

## 🔍 已知问题与解决方案

### 常见问题

#### 1. OpenAI API限制
**问题**: API调用频率限制或余额不足
**解决方案**:
```bash
# 检查API密钥和余额
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# 配置更保守的模型设置
SMART_LLM=gpt-3.5-turbo
TEMPERATURE=0.1
```

#### 2. 内存使用过高
**问题**: 长时间运行后内存占用过高
**解决方案**:
```bash
# 限制内存使用
docker run --memory=4g autogpt:latest

# 配置内存清理
MEMORY_BACKEND=redis
WIPE_REDIS_ON_START=true
```

#### 3. Docker权限问题
**问题**: 容器内无法执行某些命令
**解决方案**:
```bash
# 使用特权模式（仅开发环境）
docker run --privileged autogpt:latest

# 或配置安全的命令执行
EXECUTE_LOCAL_COMMANDS=false
SHELL_COMMAND_CONTROL=allowlist
SHELL_ALLOWLIST=ls,cat,grep
```

#### 4. 前端连接问题
**问题**: 前端无法连接到后端API
**解决方案**:
```bash
# 检查CORS配置
AP_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000

# 检查端口映射
docker run -p 8000:8000 autogpt:latest
```

### 性能优化

#### 1. 数据库优化
```sql
-- 为频繁查询的字段添加索引
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_steps_task_id ON steps(task_id);
```

#### 2. 缓存配置
```bash
# 启用Redis缓存
MEMORY_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### 3. 并发控制
```python
# 限制并发任务数量
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT=3600  # 1小时超时
```

## 🔐 安全考虑

### 安全配置

```bash
# 限制文件系统访问
RESTRICT_TO_WORKSPACE=true
ALLOW_DOWNLOADS=false

# 命令执行控制
EXECUTE_LOCAL_COMMANDS=false
SHELL_COMMAND_CONTROL=denylist
SHELL_DENYLIST=sudo,su,rm,chmod,chown

# API安全
TELEMETRY_OPT_IN=false
```

### 生产环境安全清单

- [ ] 使用强密码和API密钥
- [ ] 启用HTTPS和SSL证书
- [ ] 配置防火墙规则
- [ ] 定期更新依赖包
- [ ] 监控异常活动
- [ ] 备份重要数据

## 📈 监控与日志

### 日志配置

```bash
# 日志级别
LOG_LEVEL=INFO
LOG_FORMAT=structured_google_cloud

# 文件日志
LOG_FILE_FORMAT=simple
```

### 监控集成

```python
# Sentry错误监控
TELEMETRY_OPT_IN=true
SENTRY_DSN=your-sentry-dsn

# 自定义监控
import sentry_sdk
sentry_sdk.set_user({"id": "user_id", "email": "user@example.com"})
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 1. Fork项目并克隆
git clone https://github.com/your-username/AutoGPT.git

# 2. 创建开发分支
git checkout -b feature/your-feature

# 3. 安装开发依赖
poetry install --with dev

# 4. 设置pre-commit钩子
poetry run pre-commit install

# 5. 运行代码检查
poetry run black .
poetry run isort .
poetry run flake8
```

### 提交规范

```bash
# 提交格式
git commit -m "feat: 添加新功能描述"
git commit -m "fix: 修复bug描述"
git commit -m "docs: 更新文档"
git commit -m "test: 添加测试"
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为AutoGPT项目做出贡献的开发者和社区成员。

## 📞 支持与联系

- **官方网站**: https://agpt.co
- **文档**: https://docs.agpt.co
- **GitHub Issues**: https://github.com/Significant-Gravitas/AutoGPT/issues
- **Discord社区**: https://discord.gg/autogpt
- **邮箱支持**: support@agpt.co

---

**⚠️ 免责声明**: AutoGPT是一个实验性应用程序，按"原样"提供，不提供任何明示或暗示的保证。使用GPT-4语言模型可能因其token使用而产生费用。您有责任监控和管理自己的token使用及相关费用。