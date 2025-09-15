# AutoGPT 故障排除指南

## 概述

本文档提供了AutoGPT v0.5.1常见问题的诊断方法和解决方案，帮助开发者和运维人员快速定位和解决系统问题。

## 问题分类

### 🚨 紧急问题
- 服务完全不可用
- 数据丢失或损坏
- 安全漏洞或攻击

### ⚠️ 重要问题
- 性能严重下降
- 部分功能不可用
- 频繁错误或异常

### ℹ️ 一般问题
- 配置问题
- 使用方法问题
- 性能优化建议

## 常见问题及解决方案

### 1. 服务启动问题

#### 问题：容器启动失败

**症状**
```bash
docker: Error response from daemon: failed to create shim task
```

**诊断步骤**
```bash
# 检查Docker状态
docker info
docker system df

# 查看容器日志
docker logs autogpt-container

# 检查资源使用
docker stats
free -h
df -h
```

**解决方案**
```bash
# 清理Docker资源
docker system prune -a
docker volume prune

# 重启Docker服务
sudo systemctl restart docker

# 检查磁盘空间
sudo du -sh /var/lib/docker/*
```

#### 问题：端口冲突

**症状**
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**诊断步骤**
```bash
# 查看端口占用
netstat -tulpn | grep :8000
lsof -i :8000

# 查看进程信息
ps aux | grep autogpt
```

**解决方案**
```bash
# 方案1：杀死占用进程
sudo kill -9 <PID>

# 方案2：修改端口配置
# 编辑 docker-compose.yml
ports:
  - "8001:8000"  # 改为其他端口

# 方案3：使用环境变量
export AP_SERVER_PORT=8001
```

#### 问题：环境变量配置错误

**症状**
```
KeyError: 'OPENAI_API_KEY'
Configuration validation failed
```

**诊断步骤**
```bash
# 检查环境变量
env | grep OPENAI
docker exec autogpt-container env | grep OPENAI

# 检查配置文件
cat backend/autogpt/.env
```

**解决方案**
```bash
# 创建正确的环境配置
cp backend/autogpt/.env.example backend/autogpt/.env

# 编辑配置文件
vim backend/autogpt/.env

# 必需的环境变量
OPENAI_API_KEY=your-api-key-here
DATABASE_STRING=postgresql://user:pass@host:5432/db
REDIS_HOST=redis
REDIS_PORT=6379

# 重启服务
docker-compose restart autogpt
```

### 2. 数据库连接问题

#### 问题：PostgreSQL连接失败

**症状**
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server
```

**诊断步骤**
```bash
# 检查PostgreSQL状态
docker exec postgres pg_isready -U autogpt

# 测试连接
docker exec postgres psql -U autogpt -d autogpt -c "SELECT 1;"

# 检查网络连接
docker network ls
docker network inspect autogpt_default
```

**解决方案**
```bash
# 重启PostgreSQL
docker-compose restart postgres

# 检查数据库配置
docker exec postgres psql -U postgres -c "\l"

# 重新创建数据库
docker exec postgres createdb -U postgres autogpt

# 运行数据库迁移
docker exec autogpt-container poetry run alembic upgrade head
```

#### 问题：数据库迁移失败

**症状**
```
alembic.util.exc.CommandError: Target database is not up to date
```

**诊断步骤**
```bash
# 检查迁移状态
docker exec autogpt-container poetry run alembic current
docker exec autogpt-container poetry run alembic history

# 检查数据库表
docker exec postgres psql -U autogpt -d autogpt -c "\dt"
```

**解决方案**
```bash
# 重置数据库（注意：会丢失数据）
docker exec postgres psql -U autogpt -d autogpt -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# 重新运行迁移
docker exec autogpt-container poetry run alembic upgrade head

# 或者手动修复迁移
docker exec autogpt-container poetry run alembic stamp head
docker exec autogpt-container poetry run alembic upgrade head
```

### 3. Redis连接问题

#### 问题：Redis连接超时

**症状**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**诊断步骤**
```bash
# 检查Redis状态
docker exec redis redis-cli ping

# 检查Redis配置
docker exec redis redis-cli config get "*"

# 测试连接
docker exec autogpt-container python -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"
```

**解决方案**
```bash
# 重启Redis
docker-compose restart redis

# 清理Redis数据（如果需要）
docker exec redis redis-cli flushall

# 检查内存使用
docker exec redis redis-cli info memory
```

### 4. OpenAI API问题

#### 问题：API密钥无效

**症状**
```
openai.error.AuthenticationError: Incorrect API key provided
```

**诊断步骤**
```bash
# 检查API密钥格式
echo $OPENAI_API_KEY | wc -c  # 应该是51个字符

# 测试API密钥
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**解决方案**
```bash
# 更新API密钥
export OPENAI_API_KEY=sk-your-new-api-key

# 重启服务
docker-compose restart autogpt

# 验证配置
docker exec autogpt-container python -c "
import openai
openai.api_key = 'your-api-key'
print(openai.Model.list())
"
```

#### 问题：API配额超限

**症状**
```
openai.error.RateLimitError: You exceeded your current quota
```

**诊断步骤**
```bash
# 检查API使用情况
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/usage

# 查看应用日志中的API调用
docker logs autogpt-container | grep "openai"
```

**解决方案**
```bash
# 设置使用限制
export OPENAI_MAX_TOKENS_PER_DAY=10000
export OPENAI_MAX_REQUESTS_PER_MINUTE=60

# 使用更便宜的模型
export SMART_LLM=gpt-3.5-turbo
export FAST_LLM=gpt-3.5-turbo

# 实现请求缓存
export ENABLE_RESPONSE_CACHE=true
export CACHE_TTL=3600
```

### 5. 性能问题

#### 问题：响应时间过长

**症状**
- API响应时间超过30秒
- 任务执行缓慢
- 用户界面卡顿

**诊断步骤**
```bash
# 检查系统资源
htop
iotop
nethogs

# 检查数据库性能
docker exec postgres psql -U autogpt -d autogpt -c "
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;"

# 检查Redis性能
docker exec redis redis-cli --latency-history

# 分析应用日志
docker logs autogpt-container | grep -E "(slow|timeout|error)"
```

**解决方案**
```bash
# 优化数据库
docker exec postgres psql -U autogpt -d autogpt -c "
CREATE INDEX CONCURRENTLY idx_tasks_status ON tasks(status);
CREATE INDEX CONCURRENTLY idx_tasks_created_at ON tasks(created_at);
VACUUM ANALYZE;
"

# 增加资源限制
# 编辑 docker-compose.yml
services:
  autogpt:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

# 启用缓存
export ENABLE_MEMORY_CACHE=true
export CACHE_SIZE=1000

# 调整并发设置
export MAX_CONCURRENT_TASKS=5
export WORKER_PROCESSES=4
```

#### 问题：内存泄漏

**症状**
```
MemoryError: Unable to allocate memory
Container killed due to OOM
```

**诊断步骤**
```bash
# 监控内存使用
docker stats autogpt-container

# 检查Python内存使用
docker exec autogpt-container python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# 分析内存泄漏
docker exec autogpt-container python -m memory_profiler your_script.py
```

**解决方案**
```bash
# 设置内存限制
docker run --memory=2g autogpt:latest

# 启用垃圾回收
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# 定期重启服务
# 添加到crontab
0 2 * * * docker-compose restart autogpt
```

### 6. 网络问题

#### 问题：外部API访问失败

**症状**
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**诊断步骤**
```bash
# 测试网络连接
docker exec autogpt-container ping google.com
docker exec autogpt-container nslookup api.openai.com

# 检查防火墙设置
sudo ufw status
sudo iptables -L

# 测试HTTP连接
docker exec autogpt-container curl -I https://api.openai.com
```

**解决方案**
```bash
# 配置代理（如果需要）
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# 配置DNS
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# 检查SSL证书
docker exec autogpt-container python -c "
import ssl
import socket
context = ssl.create_default_context()
with socket.create_connection(('api.openai.com', 443)) as sock:
    with context.wrap_socket(sock, server_hostname='api.openai.com') as ssock:
        print(ssock.version())
"
```

### 7. 前端问题

#### 问题：Flutter应用无法连接后端

**症状**
```
DioError [DioErrorType.connectTimeout]: Connecting timeout
```

**诊断步骤**
```bash
# 检查后端服务状态
curl http://localhost:8000/health

# 检查网络配置
flutter doctor
flutter config

# 查看Flutter日志
flutter logs
```

**解决方案**
```bash
# 更新API端点配置
# 编辑 frontend/lib/config/api_config.dart
const String baseUrl = 'http://localhost:8000';

# 重新构建应用
flutter clean
flutter pub get
flutter run

# 配置CORS（后端）
export AP_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000
```

## 日志分析

### 日志级别和格式

```python
# 配置日志级别
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# 日志格式
export LOG_FORMAT=json  # json, simple, detailed
```

### 关键日志模式

```bash
# 错误日志
docker logs autogpt-container 2>&1 | grep -i error

# 性能日志
docker logs autogpt-container 2>&1 | grep -E "(slow|timeout|duration)"

# API调用日志
docker logs autogpt-container 2>&1 | grep -E "(openai|api_call)"

# 任务执行日志
docker logs autogpt-container 2>&1 | grep -E "(task_|execute_)"
```

### 日志聚合和分析

```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
  - add_docker_metadata:
      host: "unix:///var/run/docker.sock"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "autogpt-logs-%{+yyyy.MM.dd}"

# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [container][name] == "autogpt" {
    json {
      source => "message"
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => ["error"]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "autogpt-logs-%{+yyyy.MM.dd}"
  }
}
```

## 监控和告警

### 关键指标监控

```python
# Prometheus指标
from prometheus_client import Counter, Histogram, Gauge

# 错误率监控
ERROR_COUNTER = Counter('autogpt_errors_total', 'Total errors', ['error_type'])

# 响应时间监控
RESPONSE_TIME = Histogram('autogpt_response_time_seconds', 'Response time')

# 资源使用监控
MEMORY_USAGE = Gauge('autogpt_memory_usage_bytes', 'Memory usage')
CPU_USAGE = Gauge('autogpt_cpu_usage_percent', 'CPU usage')

# 业务指标监控
ACTIVE_TASKS = Gauge('autogpt_active_tasks', 'Number of active tasks')
TASK_SUCCESS_RATE = Histogram('autogpt_task_success_rate', 'Task success rate')
```

### 告警规则

```yaml
# prometheus-alerts.yml
groups:
- name: autogpt
  rules:
  - alert: AutoGPTHighErrorRate
    expr: rate(autogpt_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "AutoGPT error rate is high"
      description: "Error rate is {{ $value }} errors per second"

  - alert: AutoGPTHighMemoryUsage
    expr: autogpt_memory_usage_bytes / (1024*1024*1024) > 1.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "AutoGPT memory usage is high"
      description: "Memory usage is {{ $value }}GB"

  - alert: AutoGPTServiceDown
    expr: up{job="autogpt"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AutoGPT service is down"
      description: "AutoGPT service has been down for more than 1 minute"
```

## 性能调优

### 数据库优化

```sql
-- 创建必要的索引
CREATE INDEX CONCURRENTLY idx_tasks_status_created ON tasks(status, created_at);
CREATE INDEX CONCURRENTLY idx_agents_status ON agents(status);
CREATE INDEX CONCURRENTLY idx_steps_task_id ON steps(task_id);

-- 分析表统计信息
ANALYZE tasks;
ANALYZE agents;
ANALYZE steps;

-- 优化查询
EXPLAIN ANALYZE SELECT * FROM tasks WHERE status = 'running' ORDER BY created_at DESC LIMIT 10;

-- 配置参数优化
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
SELECT pg_reload_conf();
```

### 应用优化

```python
# 连接池配置
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# 异步优化
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedTaskProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.semaphore = asyncio.Semaphore(5)
    
    async def process_task(self, task):
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, 
                self._sync_process_task, 
                task
            )

# 缓存优化
from functools import lru_cache
import redis

redis_client = redis.Redis(host='redis', port=6379, db=0)

@lru_cache(maxsize=1000)
def get_cached_result(key: str):
    result = redis_client.get(key)
    if result:
        return json.loads(result)
    return None

def set_cached_result(key: str, value: dict, ttl: int = 3600):
    redis_client.setex(key, ttl, json.dumps(value))
```

## 备份和恢复

### 数据备份

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 数据库备份
docker exec postgres pg_dump -U autogpt autogpt > "$BACKUP_DIR/db_backup_$DATE.sql"

# 文件备份
docker exec autogpt-container tar -czf - /app/data > "$BACKUP_DIR/files_backup_$DATE.tar.gz"

# Redis备份
docker exec redis redis-cli --rdb - > "$BACKUP_DIR/redis_backup_$DATE.rdb"

# 上传到云存储
aws s3 cp "$BACKUP_DIR/" s3://autogpt-backups/ --recursive

# 清理旧备份（保留7天）
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +7 -delete
```

### 数据恢复

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# 停止服务
docker-compose stop autogpt

# 恢复数据库
docker exec postgres psql -U autogpt -d autogpt -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
docker exec -i postgres psql -U autogpt autogpt < "$BACKUP_FILE"

# 恢复文件
docker exec autogpt-container rm -rf /app/data
docker exec -i autogpt-container tar -xzf - -C / < "files_backup_*.tar.gz"

# 启动服务
docker-compose start autogpt

echo "Restore completed successfully"
```

## 安全问题处理

### 安全事件响应

```bash
# 1. 立即隔离受影响的服务
docker-compose stop autogpt

# 2. 收集日志和证据
docker logs autogpt-container > security_incident_$(date +%Y%m%d_%H%M%S).log
docker exec postgres pg_dump -U autogpt autogpt > security_backup_$(date +%Y%m%d_%H%M%S).sql

# 3. 分析攻击向量
grep -i "attack\|injection\|malicious" security_incident_*.log

# 4. 更新安全配置
export EXECUTE_LOCAL_COMMANDS=false
export RESTRICT_TO_WORKSPACE=true
export SHELL_COMMAND_CONTROL=denylist

# 5. 重新部署安全版本
docker-compose up -d autogpt
```

### 安全加固检查清单

- [ ] 更新所有依赖到最新版本
- [ ] 启用HTTPS和SSL证书
- [ ] 配置防火墙规则
- [ ] 启用访问日志和审计
- [ ] 设置强密码策略
- [ ] 启用多因素认证
- [ ] 定期安全扫描
- [ ] 备份加密存储

## 联系支持

### 问题报告模板

```markdown
## 问题描述
简要描述遇到的问题

## 环境信息
- AutoGPT版本: v0.5.1
- 操作系统: Ubuntu 20.04
- Docker版本: 20.10.21
- Python版本: 3.10.12

## 重现步骤
1. 步骤1
2. 步骤2
3. 步骤3

## 期望结果
描述期望的正常行为

## 实际结果
描述实际发生的情况

## 错误日志
```
粘贴相关的错误日志
```

## 已尝试的解决方案
列出已经尝试过的解决方法

## 附加信息
其他可能有用的信息
```

### 获取帮助的渠道

- **GitHub Issues**: https://github.com/Significant-Gravitas/AutoGPT/issues
- **Discord社区**: https://discord.gg/autogpt
- **文档网站**: https://docs.agpt.co/
- **Stack Overflow**: 标签 `autogpt`

### 紧急支持

对于生产环境的紧急问题：
1. 立即隔离问题服务
2. 收集必要的日志和信息
3. 通过官方渠道报告问题
4. 实施临时解决方案
5. 等待官方修复或指导

这个故障排除指南涵盖了AutoGPT系统可能遇到的各种问题和解决方案，帮助用户快速定位和解决问题。