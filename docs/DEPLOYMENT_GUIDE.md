# AutoGPT 部署指南

## 概述

本指南详细介绍了AutoGPT v0.5.1在不同环境中的部署方法，包括开发环境、测试环境和生产环境的完整部署流程。

## 部署架构

### 系统组件

```mermaid
graph TB
    subgraph "负载均衡层"
        A[Nginx/HAProxy] --> B[SSL终端]
    end
    
    subgraph "应用层"
        B --> C[AutoGPT Server 1]
        B --> D[AutoGPT Server 2]
        B --> E[AutoGPT Server N]
    end
    
    subgraph "数据层"
        C --> F[PostgreSQL主库]
        D --> F
        E --> F
        F --> G[PostgreSQL从库]
        
        C --> H[Redis缓存]
        D --> H
        E --> H
        
        C --> I[文件存储]
        D --> I
        E --> I
    end
    
    subgraph "外部服务"
        C --> J[OpenAI API]
        D --> J
        E --> J
        
        C --> K[监控服务]
        D --> K
        E --> K
    end
```

## 环境要求

### 硬件要求

#### 开发环境
- **CPU**: 2核心以上
- **内存**: 4GB RAM
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接

#### 生产环境
- **CPU**: 4核心以上（推荐8核心）
- **内存**: 8GB RAM（推荐16GB）
- **存储**: 100GB SSD（推荐500GB）
- **网络**: 高速稳定的互联网连接
- **带宽**: 100Mbps以上

### 软件要求

#### 基础软件
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / macOS 11+ / Windows 10+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.10+（本地开发）
- **Node.js**: 16+（前端开发）

#### 数据库
- **PostgreSQL**: 13+（推荐）
- **Redis**: 6+（缓存和会话存储）
- **SQLite**: 3.35+（开发环境）

## 开发环境部署

### 方式一：Docker Compose（推荐）

#### 1. 准备工作

```bash
# 克隆项目
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd AutoGPT

# 创建环境配置文件
cp backend/autogpt/.env.example backend/autogpt/.env
```

#### 2. 配置环境变量

编辑 `backend/autogpt/.env` 文件：

```bash
# OpenAI配置
OPENAI_API_KEY=your-openai-api-key-here
SMART_LLM=gpt-4-turbo
FAST_LLM=gpt-3.5-turbo
TEMPERATURE=0.1

# 数据库配置
DATABASE_STRING=postgresql://autogpt:password@postgres:5432/autogpt
REDIS_HOST=redis
REDIS_PORT=6379

# 服务器配置
AP_SERVER_PORT=8000
AP_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000

# 安全配置
EXECUTE_LOCAL_COMMANDS=false
RESTRICT_TO_WORKSPACE=true
SHELL_COMMAND_CONTROL=denylist
SHELL_DENYLIST=sudo,su,rm,chmod,chown

# 日志配置
LOG_LEVEL=DEBUG
LOG_FORMAT=simple
PLAIN_OUTPUT=false
```

#### 3. 创建Docker Compose配置

创建 `docker-compose.dev.yml`：

```yaml
version: '3.9'

services:
  autogpt:
    build: 
      context: ./backend/autogpt
      dockerfile: Dockerfile
      args:
        BUILD_TYPE: dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_STRING=postgresql://autogpt:password@postgres:5432/autogpt
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    env_file:
      - ./backend/autogpt/.env
    volumes:
      - ./backend/autogpt:/app
      - autogpt-data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: autogpt
      POSTGRES_USER: autogpt
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - autogpt
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  autogpt-data:
```

#### 4. 启动服务

```bash
# 启动所有服务
docker-compose -f docker-compose.dev.yml up -d

# 查看日志
docker-compose -f docker-compose.dev.yml logs -f autogpt

# 停止服务
docker-compose -f docker-compose.dev.yml down
```

### 方式二：本地开发

#### 1. 安装依赖

```bash
# 安装Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 安装后端依赖
cd backend/autogpt
poetry install --with dev

# 安装前端依赖
cd ../../frontend
flutter pub get
```

#### 2. 启动数据库服务

```bash
# 使用Docker启动数据库
docker run -d --name postgres \
  -e POSTGRES_DB=autogpt \
  -e POSTGRES_USER=autogpt \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

#### 3. 启动应用

```bash
# 启动后端（终端1）
cd backend/autogpt
poetry run autogpt serve

# 启动前端（终端2）
cd frontend
flutter run -d chrome --web-port 3000
```

## 测试环境部署

### 1. 准备测试环境

```bash
# 创建测试配置
cp backend/autogpt/.env.example backend/autogpt/.env.test

# 编辑测试配置
vim backend/autogpt/.env.test
```

测试环境配置：

```bash
# 使用测试API密钥
OPENAI_API_KEY=test-api-key

# 测试数据库
DATABASE_STRING=postgresql://test_user:test_pass@test-db:5432/autogpt_test

# 启用详细日志
LOG_LEVEL=DEBUG
TELEMETRY_OPT_IN=true

# 测试模式配置
EXECUTE_LOCAL_COMMANDS=true
ALLOW_DOWNLOADS=true
```

### 2. 创建测试Docker Compose

创建 `docker-compose.test.yml`：

```yaml
version: '3.9'

services:
  autogpt-test:
    build: 
      context: ./backend/autogpt
      dockerfile: Dockerfile
      args:
        BUILD_TYPE: dev
    environment:
      - DATABASE_STRING=postgresql://test_user:test_pass@test-postgres:5432/autogpt_test
      - REDIS_HOST=test-redis
    env_file:
      - ./backend/autogpt/.env.test
    command: ["poetry", "run", "pytest", "-v", "--cov=autogpt"]
    volumes:
      - ./backend/autogpt:/app
      - test-reports:/app/reports
    depends_on:
      - test-postgres
      - test-redis

  test-postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: autogpt_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
    tmpfs:
      - /var/lib/postgresql/data

  test-redis:
    image: redis:7-alpine
    tmpfs:
      - /data

volumes:
  test-reports:
```

### 3. 运行测试

```bash
# 运行所有测试
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# 运行特定测试
docker-compose -f docker-compose.test.yml run autogpt-test poetry run pytest tests/unit -v

# 生成覆盖率报告
docker-compose -f docker-compose.test.yml run autogpt-test poetry run pytest --cov=autogpt --cov-report=html
```

## 生产环境部署

### 方式一：Docker Swarm部署

#### 1. 准备生产环境

```bash
# 初始化Docker Swarm
docker swarm init

# 创建网络
docker network create --driver overlay autogpt-network

# 创建secrets
echo "your-openai-api-key" | docker secret create openai_api_key -
echo "your-db-password" | docker secret create db_password -
```

#### 2. 创建生产配置

创建 `docker-compose.prod.yml`：

```yaml
version: '3.9'

services:
  autogpt:
    image: autogpt:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    environment:
      - DATABASE_STRING=postgresql://autogpt:password@postgres:5432/autogpt
      - REDIS_HOST=redis
      - LOG_LEVEL=INFO
      - TELEMETRY_OPT_IN=true
    secrets:
      - openai_api_key
    networks:
      - autogpt-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
    environment:
      POSTGRES_DB: autogpt
      POSTGRES_USER: autogpt
    secrets:
      - source: db_password
        target: POSTGRES_PASSWORD
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - autogpt-network

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
    volumes:
      - redis-data:/data
    networks:
      - autogpt-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    networks:
      - autogpt-network
    depends_on:
      - autogpt

volumes:
  postgres-data:
  redis-data:

networks:
  autogpt-network:
    external: true

secrets:
  openai_api_key:
    external: true
  db_password:
    external: true
```

#### 3. 配置Nginx

创建 `nginx.conf`：

```nginx
events {
    worker_connections 1024;
}

http {
    upstream autogpt_backend {
        server autogpt:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        client_max_body_size 100M;

        location / {
            proxy_pass http://autogpt_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

#### 4. 部署到Swarm

```bash
# 构建镜像
docker build -t autogpt:latest --build-arg BUILD_TYPE=release ./backend/autogpt

# 部署服务
docker stack deploy -c docker-compose.prod.yml autogpt

# 查看服务状态
docker service ls
docker service logs autogpt_autogpt
```

### 方式二：Kubernetes部署

#### 1. 创建命名空间

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autogpt
```

#### 2. 创建ConfigMap和Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autogpt-config
  namespace: autogpt
data:
  DATABASE_STRING: "postgresql://autogpt:password@postgres:5432/autogpt"
  REDIS_HOST: "redis"
  LOG_LEVEL: "INFO"
  AP_SERVER_PORT: "8000"

---
apiVersion: v1
kind: Secret
metadata:
  name: autogpt-secrets
  namespace: autogpt
type: Opaque
data:
  openai-api-key: <base64-encoded-api-key>
  db-password: <base64-encoded-password>
```

#### 3. 创建部署配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogpt
  namespace: autogpt
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autogpt
  template:
    metadata:
      labels:
        app: autogpt
    spec:
      containers:
      - name: autogpt
        image: autogpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: openai-api-key
        envFrom:
        - configMapRef:
            name: autogpt-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: autogpt-service
  namespace: autogpt
spec:
  selector:
    app: autogpt
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 4. 创建Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autogpt-ingress
  namespace: autogpt
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: autogpt-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autogpt-service
            port:
              number: 80
```

#### 5. 部署到Kubernetes

```bash
# 应用配置
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml

# 查看状态
kubectl get pods -n autogpt
kubectl get services -n autogpt
kubectl logs -f deployment/autogpt -n autogpt
```

## 云平台部署

### AWS部署

#### 1. 使用ECS Fargate

```json
{
  "family": "autogpt-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "autogpt",
      "image": "your-account.dkr.ecr.region.amazonaws.com/autogpt:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_STRING",
          "value": "postgresql://user:pass@rds-endpoint:5432/autogpt"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/autogpt",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 2. 使用Terraform部署

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_ecs_cluster" "autogpt" {
  name = "autogpt-cluster"
}

resource "aws_ecs_service" "autogpt" {
  name            = "autogpt-service"
  cluster         = aws_ecs_cluster.autogpt.id
  task_definition = aws_ecs_task_definition.autogpt.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.autogpt.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.autogpt.arn
    container_name   = "autogpt"
    container_port   = 8000
  }
}
```

### Google Cloud Platform部署

#### 1. 使用Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: autogpt
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/project-id/autogpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

部署命令：

```bash
# 构建并推送镜像
gcloud builds submit --tag gcr.io/project-id/autogpt:latest

# 部署到Cloud Run
gcloud run deploy autogpt \
  --image gcr.io/project-id/autogpt:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 监控和维护

### 1. 健康检查

创建健康检查端点：

```python
# health.py
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.5.1"
    }

@router.get("/ready")
async def readiness_check():
    # 检查数据库连接、Redis连接等
    return {"status": "ready"}
```

### 2. 日志聚合

使用ELK Stack或类似工具：

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
```

### 3. 监控指标

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter('autogpt_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('autogpt_request_duration_seconds', 'Request duration')
ACTIVE_TASKS = Gauge('autogpt_active_tasks', 'Number of active tasks')
```

### 4. 备份策略

```bash
#!/bin/bash
# backup.sh

# 数据库备份
pg_dump -h postgres -U autogpt autogpt > backup_$(date +%Y%m%d_%H%M%S).sql

# 文件备份
tar -czf files_backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/data

# 上传到云存储
aws s3 cp backup_*.sql s3://autogpt-backups/
aws s3 cp files_backup_*.tar.gz s3://autogpt-backups/
```

## 故障排除

### 常见问题

#### 1. 容器启动失败

```bash
# 查看容器日志
docker logs autogpt-container

# 检查资源使用
docker stats

# 进入容器调试
docker exec -it autogpt-container /bin/bash
```

#### 2. 数据库连接问题

```bash
# 测试数据库连接
docker exec -it postgres psql -U autogpt -d autogpt -c "SELECT 1;"

# 检查网络连接
docker network ls
docker network inspect autogpt-network
```

#### 3. 性能问题

```bash
# 监控资源使用
htop
iotop
nethogs

# 分析慢查询
tail -f /var/log/postgresql/postgresql.log | grep "slow query"
```

### 性能调优

#### 1. 数据库优化

```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_tasks_status ON tasks(status);
CREATE INDEX CONCURRENTLY idx_tasks_created_at ON tasks(created_at);

-- 分析查询性能
EXPLAIN ANALYZE SELECT * FROM tasks WHERE status = 'running';
```

#### 2. 应用优化

```python
# 连接池配置
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# 缓存配置
REDIS_CONNECTION_POOL_SIZE = 50
CACHE_TTL = 3600
```

#### 3. 系统优化

```bash
# 调整系统参数
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
echo 'fs.file-max=65536' >> /etc/sysctl.conf
sysctl -p
```

## 安全加固

### 1. 网络安全

```bash
# 防火墙配置
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # 只允许内部访问
ufw enable
```

### 2. 容器安全

```dockerfile
# 使用非root用户
RUN adduser --disabled-password --gecos '' autogpt
USER autogpt

# 只读文件系统
docker run --read-only --tmpfs /tmp autogpt:latest
```

### 3. 密钥管理

```bash
# 使用Docker secrets
echo "api-key" | docker secret create openai_api_key -

# 或使用外部密钥管理
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id openai-key --query SecretString --output text)
```

这个部署指南涵盖了从开发到生产的完整部署流程，包括监控、维护和故障排除。根据实际需求选择合适的部署方式。