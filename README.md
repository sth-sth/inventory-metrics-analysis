# Inventory Intelligence Cloud / 库存智能决策云

Production-ready inventory analytics web app with bilingual UX and decision support.
面向生产环境的库存分析网页应用，支持中英文界面与智能决策支持。

## 1. Quick Start / 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Core Capabilities / 核心能力

- Bilingual UI switch (Chinese/English) / 中英文界面切换
- Demo + upload data modes / 演示数据与上传数据双模式
- KPI automation and alerting / 自动 KPI 与风险报警
- Evidence-based attribution / 基于证据的归因分析
- Smart scenario simulation / 智能场景模拟
- Budget-constrained replenishment plan / 预算约束下的补货决策
- One-click CSV exports for decisions / 一键导出决策清单

## 3. Input Modes / 数据输入模式

### Demo Mode / 演示模式

Uses built-in files directly:
直接使用内置文件：

- data/inventory_demo.csv
- data/transactions_demo.csv

### Upload Mode / 上传模式

Upload both files:
需上传两份文件：

- Inventory snapshot CSV / 库存快照 CSV
- Transactions CSV / 交易流水 CSV

## 4. Required Schema / 必要字段

### Inventory CSV

- date
- sku
- category
- warehouse
- on_hand_qty
- avg_daily_demand
- lead_time_days
- unit_cost

### Transactions CSV

- date
- sku
- warehouse
- event_type (`sale` | `receipt` | `adjustment`)
- qty
- delay_days
- supplier

## 5. Formula Transparency / 公式透明化（细粒度评分）

### Core formulas / 核心公式

- DOH = on_hand_qty / avg_daily_demand
- lead_time_demand = avg_daily_demand * lead_time_days
- safety_stock = z * std(avg_daily_demand) * sqrt(lead_time_days)
- reorder_point = lead_time_demand + safety_stock
- coverage_gap = on_hand_qty - reorder_point

### Granular scoring / 细粒度评分

- Demand forecast error = |actual demand - forecast demand| / forecast demand, normalized to 0-1
- Supplier delay score = avg_delay_days / 7, normalized to 0-1
- ROP setting score = |coverage_gap| / reorder_point, normalized to 0-1
- Info sync score = info_sync_delay_days / 5, normalized to 0-1
- Domain score = mean of the factors in that domain
- Overall score = mean of all factors across four domains

### Replenishment base qty / 补货基础量

- base_order_qty = max(reorder_point - on_hand_qty, 0)

## 6. Smarter Decision Support / 更智能的决策支持

The app now includes:
当前已增强：

- Warehouse/category filters / 仓库与品类筛选
- Downloadable alerts/recommendations/plan / 报警、建议、补货清单可下载
- Scenario simulation for demand and lead time shifts / 需求与交期变化情景模拟
- Budget-constrained prioritization / 预算约束下优先级排序
- Target service-level aware order uplift / 服务水平目标驱动补货调整

## 7. Further Improvement Opportunities / 进一步改进方向

- Add role-based dashboards (CEO/planner/buyer) / 增加角色化看板
- Add workflow ownership + SLA loop / 增加责任人和 SLA 闭环
- Integrate external signals (promotion/season/weather) / 引入促销季节天气等外部信号
- Add probabilistic demand forecast / 增加概率预测与置信区间
- Add optimization solver for multi-constraint planning / 引入多约束优化求解器
- Add anomaly detection on transaction stream / 增加流水异常检测

## 8. Deployment / 部署

See [docs/deployment.md](docs/deployment.md) for Streamlit Cloud, Render, and Docker options.
Streamlit Cloud、Render、Docker 部署步骤见 [docs/deployment.md](docs/deployment.md)。

Included deployment assets / 已包含部署文件：

- Dockerfile
- .dockerignore
- .streamlit/config.toml
- render.yaml
- .github/workflows/docker-image.yml

## 9. References / 参考

- [docs/industry_playbook.md](docs/industry_playbook.md)
- [docs/methodology_cn_en.md](docs/methodology_cn_en.md)
