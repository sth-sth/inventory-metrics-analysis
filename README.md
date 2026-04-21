# Inventory Intelligence Cloud

A production-ready web inventory analytics app with:
- Modern UI/UX dashboard (executive + operational views)
- Built-in demo mode for instant showcase
- CSV upload mode for real business data
- Automatic KPI computation and risk monitoring
- Alert center (stockout, overstock, supplier delay)
- ABC classification
- Evidence-based root-cause attribution with confidence
- Data-driven business recommendations

## 1. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. UI/UX Layout

- Hero section and KPI cards for executive snapshot
- Tabbed flow:
	- Executive Overview
	- Risk Monitoring
	- Root-Cause Diagnosis
	- Action Center
- Sidebar control panel:
	- Data source mode (`built-in demo` or `upload`)
	- Threshold customization

This structure supports both high-level management reading and low-level operational drill-down.

## 3. Data Input Modes

### Built-in Demo Mode

No upload required. The app auto-loads:
- data/inventory_demo.csv
- data/transactions_demo.csv

### Upload Mode

Upload both files in sidebar:
- Inventory snapshot CSV
- Transactions CSV

## 4. Required CSV Schema

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

## 5. Method Notes (No Arbitrary Inference)

Attribution is derived from observed deviations vs benchmark:
- Demand pressure: sales minus receipts
- Delay pressure: average receipt delay
- Planning pressure: stock coverage gap

Outputs include:
- `impact_score`: relative factor contribution
- `confidence`: signal confidence from data coverage

This is evidence-based diagnostic attribution, not strict causal proof.

## 6. Deploy to Production Web

See deployment guide:
- docs/deployment.md

Included deployment assets:
- Dockerfile
- .dockerignore
- .streamlit/config.toml
- render.yaml
- .github/workflows/docker-image.yml

## 7. Professional Baseline

Industry and governance references:
- docs/industry_playbook.md
