# Monitoring Dashboards

Static dashboards for visualising AutoGPT metrics such as success rates,
bottlenecks and blueprint versions. The charts expect a backend API that
serves JSON data from the monitoring storage module.

To view the dashboard locally run any static web server:

```bash
python -m http.server --directory frontend/monitoring
```
