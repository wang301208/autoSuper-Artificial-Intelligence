# AutoGPT API 参考文档

## 概述

AutoGPT v0.5.1 实现了标准的 Agent Protocol 规范，提供完整的 RESTful API 接口用于智能体管理、任务执行和文件操作。

## 基础信息

- **基础URL**: `http://localhost:8000/ap/v1/agent`
- **API版本**: v1
- **认证方式**: 暂不需要认证（开发环境）
- **内容类型**: `application/json`
- **字符编码**: UTF-8

## 通用响应格式

### 成功响应
```json
{
  "success": true,
  "data": {},
  "message": "操作成功"
}
```

### 错误响应
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {}
  }
}
```

### HTTP状态码
- `200` - 请求成功
- `201` - 资源创建成功
- `400` - 请求参数错误
- `404` - 资源不存在
- `500` - 服务器内部错误

## 任务管理 API

### 创建任务

创建一个新的智能体任务。

**请求**
```http
POST /ap/v1/agent/tasks
Content-Type: application/json

{
  "input": "请分析这个销售数据文件并生成月度报告",
  "additional_input": {
    "priority": "high",
    "deadline": "2024-01-31T23:59:59Z",
    "context": {
      "department": "sales",
      "region": "asia-pacific"
    }
  }
}
```

**响应**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "input": "请分析这个销售数据文件并生成月度报告",
  "additional_input": {
    "priority": "high",
    "deadline": "2024-01-31T23:59:59Z",
    "context": {
      "department": "sales",
      "region": "asia-pacific"
    }
  },
  "status": "created",
  "created_at": "2024-01-15T10:30:00Z",
  "modified_at": "2024-01-15T10:30:00Z",
  "artifacts": []
}
```

### 获取任务列表

获取所有任务的列表，支持分页和过滤。

**请求**
```http
GET /ap/v1/agent/tasks?page=1&size=10&status=running
```

**查询参数**
- `page` (可选): 页码，默认为1
- `size` (可选): 每页大小，默认为10，最大100
- `status` (可选): 任务状态过滤 (`created`, `running`, `completed`, `failed`)
- `created_after` (可选): 创建时间过滤 (ISO 8601格式)

**响应**
```json
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "input": "请分析这个销售数据文件并生成月度报告",
      "status": "running",
      "created_at": "2024-01-15T10:30:00Z",
      "modified_at": "2024-01-15T10:35:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "size": 10,
    "total": 25,
    "pages": 3
  }
}
```

### 获取任务详情

获取指定任务的详细信息。

**请求**
```http
GET /ap/v1/agent/tasks/{task_id}
```

**响应**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "input": "请分析这个销售数据文件并生成月度报告",
  "additional_input": {
    "priority": "high",
    "deadline": "2024-01-31T23:59:59Z"
  },
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "modified_at": "2024-01-15T11:45:00Z",
  "artifacts": [
    {
      "artifact_id": "artifact-123",
      "file_name": "sales_report_january.pdf",
      "relative_path": "reports/sales_report_january.pdf",
      "created_at": "2024-01-15T11:45:00Z"
    }
  ]
}
```

### 删除任务

删除指定的任务及其相关数据。

**请求**
```http
DELETE /ap/v1/agent/tasks/{task_id}
```

**响应**
```json
{
  "success": true,
  "message": "任务已成功删除"
}
```

## 步骤执行 API

### 执行步骤

为指定任务执行下一个步骤。

**请求**
```http
POST /ap/v1/agent/tasks/{task_id}/steps
Content-Type: application/json

{
  "input": "请继续分析数据，重点关注Q4的销售趋势",
  "additional_input": {
    "focus_area": "q4_trends",
    "include_charts": true
  }
}
```

**响应**
```json
{
  "step_id": "step-456",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "analyze_sales_data",
  "status": "completed",
  "input": "请继续分析数据，重点关注Q4的销售趋势",
  "additional_input": {
    "focus_area": "q4_trends",
    "include_charts": true
  },
  "output": "已完成Q4销售数据分析，发现销售额较Q3增长15%。主要增长来源于亚太地区的企业客户。已生成相关图表。",
  "artifacts": [
    {
      "artifact_id": "chart-789",
      "file_name": "q4_sales_chart.png",
      "relative_path": "charts/q4_sales_chart.png"
    }
  ],
  "is_last": false,
  "created_at": "2024-01-15T11:30:00Z"
}
```

### 获取步骤列表

获取指定任务的所有执行步骤。

**请求**
```http
GET /ap/v1/agent/tasks/{task_id}/steps?page=1&size=20
```

**响应**
```json
{
  "steps": [
    {
      "step_id": "step-456",
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "analyze_sales_data",
      "status": "completed",
      "output": "已完成Q4销售数据分析...",
      "created_at": "2024-01-15T11:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "size": 20,
    "total": 5,
    "pages": 1
  }
}
```

### 获取步骤详情

获取指定步骤的详细信息。

**请求**
```http
GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}
```

**响应**
```json
{
  "step_id": "step-456",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "analyze_sales_data",
  "status": "completed",
  "input": "请继续分析数据，重点关注Q4的销售趋势",
  "additional_input": {
    "focus_area": "q4_trends",
    "include_charts": true
  },
  "output": "已完成Q4销售数据分析，发现销售额较Q3增长15%。主要增长来源于亚太地区的企业客户。已生成相关图表。",
  "artifacts": [
    {
      "artifact_id": "chart-789",
      "file_name": "q4_sales_chart.png",
      "relative_path": "charts/q4_sales_chart.png",
      "created_at": "2024-01-15T11:30:00Z"
    }
  ],
  "is_last": false,
  "created_at": "2024-01-15T11:30:00Z"
}
```

## 文件管理 API

### 上传文件

为指定任务上传文件。

**请求**
```http
POST /ap/v1/agent/tasks/{task_id}/artifacts
Content-Type: multipart/form-data

file: (binary data)
relative_path: data/sales_data.csv
```

**响应**
```json
{
  "artifact_id": "artifact-123",
  "file_name": "sales_data.csv",
  "relative_path": "data/sales_data.csv",
  "file_size": 1024000,
  "created_at": "2024-01-15T10:35:00Z"
}
```

### 下载文件

下载指定的文件。

**请求**
```http
GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}
```

**响应**
```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="sales_data.csv"
Content-Length: 1024000

(binary file data)
```

### 获取文件列表

获取指定任务的所有文件。

**请求**
```http
GET /ap/v1/agent/tasks/{task_id}/artifacts
```

**响应**
```json
{
  "artifacts": [
    {
      "artifact_id": "artifact-123",
      "file_name": "sales_data.csv",
      "relative_path": "data/sales_data.csv",
      "file_size": 1024000,
      "created_at": "2024-01-15T10:35:00Z"
    },
    {
      "artifact_id": "artifact-456",
      "file_name": "sales_report.pdf",
      "relative_path": "reports/sales_report.pdf",
      "file_size": 2048000,
      "created_at": "2024-01-15T11:45:00Z"
    }
  ]
}
```

### 删除文件

删除指定的文件。

**请求**
```http
DELETE /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}
```

**响应**
```json
{
  "success": true,
  "message": "文件已成功删除"
}
```

## 智能体管理 API

### 获取智能体状态

获取当前智能体的状态信息。

**请求**
```http
GET /ap/v1/agent/status
```

**响应**
```json
{
  "agent_id": "agent-main",
  "status": "running",
  "version": "0.5.1",
  "capabilities": [
    "file_operations",
    "web_browsing",
    "code_execution",
    "data_analysis"
  ],
  "active_tasks": 3,
  "total_tasks": 25,
  "uptime": "2024-01-15T08:00:00Z",
  "memory_usage": {
    "used": "512MB",
    "total": "2GB"
  },
  "api_usage": {
    "openai_tokens_used": 15000,
    "openai_cost": 0.45
  }
}
```

### 重启智能体

重启智能体服务。

**请求**
```http
POST /ap/v1/agent/restart
```

**响应**
```json
{
  "success": true,
  "message": "智能体正在重启",
  "estimated_downtime": "30s"
}
```

## WebSocket API

### 实时任务状态

通过WebSocket获取任务执行的实时状态更新。

**连接**
```javascript
const ws = new WebSocket('ws://localhost:8000/ap/v1/agent/tasks/{task_id}/ws');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('任务状态更新:', data);
};
```

**消息格式**
```json
{
  "type": "status_update",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.6,
  "current_step": "生成报告",
  "timestamp": "2024-01-15T11:30:00Z"
}
```

### 实时日志流

获取智能体执行的实时日志。

**连接**
```javascript
const ws = new WebSocket('ws://localhost:8000/ap/v1/agent/logs/ws');
```

**消息格式**
```json
{
  "type": "log",
  "level": "INFO",
  "message": "开始执行文件分析任务",
  "timestamp": "2024-01-15T11:30:00Z",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "component": "file_analyzer"
}
```

## 错误处理

### 错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| `INVALID_REQUEST` | 400 | 请求参数无效 |
| `TASK_NOT_FOUND` | 404 | 任务不存在 |
| `STEP_NOT_FOUND` | 404 | 步骤不存在 |
| `ARTIFACT_NOT_FOUND` | 404 | 文件不存在 |
| `TASK_ALREADY_RUNNING` | 409 | 任务已在运行中 |
| `INSUFFICIENT_RESOURCES` | 429 | 资源不足 |
| `OPENAI_API_ERROR` | 502 | OpenAI API错误 |
| `INTERNAL_ERROR` | 500 | 服务器内部错误 |

### 错误响应示例

```json
{
  "success": false,
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "指定的任务不存在",
    "details": {
      "task_id": "invalid-task-id",
      "suggestion": "请检查任务ID是否正确"
    }
  }
}
```

## 限制和配额

### API限制
- **请求频率**: 每分钟最多100个请求
- **文件上传**: 单个文件最大100MB
- **并发任务**: 最多同时运行5个任务
- **任务超时**: 单个任务最长执行时间1小时

### OpenAI API配额
- **Token限制**: 根据OpenAI账户配额
- **成本控制**: 可通过环境变量设置预算限制
- **模型选择**: 支持GPT-3.5和GPT-4模型

## SDK和客户端库

### Python客户端

```python
from autogpt_client import AutoGPTClient

client = AutoGPTClient(base_url="http://localhost:8000")

# 创建任务
task = client.create_task(
    input="分析销售数据",
    additional_input={"priority": "high"}
)

# 执行步骤
step = client.execute_step(
    task_id=task.task_id,
    input="开始分析"
)

# 获取结果
artifacts = client.get_artifacts(task.task_id)
```

### JavaScript客户端

```javascript
import { AutoGPTClient } from '@autogpt/client';

const client = new AutoGPTClient({
  baseURL: 'http://localhost:8000'
});

// 创建任务
const task = await client.createTask({
  input: '分析销售数据',
  additionalInput: { priority: 'high' }
});

// 执行步骤
const step = await client.executeStep(task.taskId, {
  input: '开始分析'
});
```

## 最佳实践

### 1. 任务设计
- 将复杂任务分解为多个简单步骤
- 提供清晰的任务描述和上下文
- 设置合理的优先级和截止时间

### 2. 错误处理
- 实现重试机制处理临时错误
- 监控API配额和成本
- 记录详细的错误日志

### 3. 性能优化
- 使用分页获取大量数据
- 缓存频繁访问的数据
- 合理设置请求超时时间

### 4. 安全考虑
- 验证所有输入参数
- 限制文件上传类型和大小
- 监控异常API调用模式

## 更新日志

### v0.5.1
- 新增多模态输入支持
- 改进错误处理和响应格式
- 添加WebSocket实时通信
- 增强文件管理功能

### v0.5.0
- 初始API版本发布
- 实现Agent Protocol规范
- 基础任务和步骤管理功能