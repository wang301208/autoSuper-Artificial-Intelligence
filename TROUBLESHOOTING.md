# 故障排除指南

本页面列出了使用 AutoGPT 时可能遇到的常见问题及其解决方法。

## 🔧 安装问题

### Poetry 配置错误
**问题**: Poetry 配置无效，出现 "不允许额外的属性" 错误

**解决方案**:
```bash
# 更新 Poetry 到最新版本
pip install --upgrade poetry

# 清理缓存
poetry cache clear pypi --all
```

### 依赖安装失败
**问题**: 安装依赖时出现版本冲突

**解决方案**:
```bash
# 删除虚拟环境重新安装
poetry env remove python
poetry install
```

## 🤖 代理运行问题

### Pydantic 验证错误
**问题**: 启动代理时出现数据验证错误

**解决方案**:
```bash
# 删除旧的数据库文件
rm -f agent.db
rm -rf workspace/
```

### API 密钥错误
**问题**: OpenAI API 密钥无效或未设置

**解决方案**:
1. 检查 `.env` 文件中的 `OPENAI_API_KEY`
2. 确保 API 密钥有效且有足够余额
3. 重启代理服务

## 🌐 网络问题

### 端口占用
**问题**: 8000 端口被占用

**解决方案**:
```bash
# 查找占用端口的进程
lsof -i :8000

# 终止进程
kill -9 <PID>
```

## 💬 获取帮助

如果以上解决方案无法解决你的问题：

1. **查看日志**: 检查详细的错误信息
2. **搜索 Issues**: [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues)
3. **社区支持**: [Discord](https://discord.gg/autogpt)
4. **创建新 Issue**: 提供详细的错误信息和复现步骤