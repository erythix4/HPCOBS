# AI Observability for HPC: Beyond GPU Utilization

**FOSDEM 2026 - HPC Devroom Demo**  
Samuel Desseaux - Erythix / Aureonis

---

## The Problem

> Your GPU is at 95% utilization. Great. But is your model actually learning?

Most HPC teams monitor infrastructure metrics: GPU utilization, memory, power.
These can look **perfectly healthy** while your training is broken.

**High utilization ≠ useful work**

## The Solution: 3-Layer Observability

| Layer | Focus | Key Metrics |
|-------|-------|-------------|
| **Infrastructure** | Hardware health | GPU util, VRAM, power, temp |
| **Workload** | Training efficiency | Throughput, I/O ratio, MFU |
| **Model Health** | Learning progress | Loss, gradients, convergence |

**Cross-layer correlation reveals hidden problems.**

---

## Quick Start

```bash
# Start the demo
docker-compose up -d

# Open the landing page
open http://localhost:8080

# Or go directly to Grafana
open http://localhost:3000
```

## Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Landing Page** | http://localhost:8080 | Observability explainer |
| **Grafana** | http://localhost:3000 | Metrics dashboards |
| **VictoriaMetrics** | http://localhost:8428 | Time-series database |
| **VMAlert** | http://localhost:8880 | Alerting engine |
| **Training Simulator** | http://localhost:9101/metrics | Prometheus metrics |

Grafana login: `admin` / `fosdem2026`

---

## Demo Scenarios

Switch scenarios to see how 3-layer observability catches problems that traditional monitoring misses.

### 1. Normal (baseline)
```bash
SCENARIO=normal docker-compose up -d
```
All metrics healthy. Use as baseline for comparison.

### 2. I/O Bottleneck (hidden problem)
```bash
SCENARIO=io_bottleneck docker-compose up -d
```
- **Layer 1**: GPU util ~92% (looks fine!)
- **Layer 2**: Throughput drops 40%, data loading ratio rises
- **Layer 3**: Loss still decreasing (slowly)

**Without Layer 2, you'd debug for hours.**

### 3. Gradient Explosion
```bash
SCENARIO=gradient_explosion docker-compose up -d
```
- **Layer 1**: All metrics normal
- **Layer 2**: Throughput normal
- **Layer 3**: Gradient norm explodes >100, loss increases

**Layer 3 catches this before it crashes.**

### 4. Loss Plateau
```bash
SCENARIO=loss_plateau docker-compose up -d
```
- **Layer 1**: GPU util ~87%
- **Layer 2**: Throughput normal
- **Layer 3**: Loss stagnates at 1.15, vanishing gradients

**Model stopped learning despite everything running.**

---

## Key Metrics Reference

### Layer 1: Infrastructure (DCGM-style)
| Metric | Description |
|--------|-------------|
| `dcgm_fi_dev_gpu_util` | GPU core utilization % |
| `dcgm_fi_dev_tensor_active` | Tensor core utilization % |
| `dcgm_fi_dev_fb_used` | VRAM usage in MB |
| `dcgm_fi_dev_power_usage` | Power draw in watts |
| `dcgm_fi_dev_nvlink_bandwidth_total` | NVLink bandwidth GB/s |

### Layer 2: Workload Efficiency
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `training_samples_per_second` | Real throughput | Drop >40% |
| `training_data_loading_ratio` | I/O wait time | >15% |
| `training_model_flops_utilization` | MFU (gold standard) | <30% |
| `training_all_reduce_time_seconds` | Gradient sync overhead | >200ms |
| `training_batch_efficiency` | Effective batch utilization | <70% |

### Layer 3: Model Health
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `training_loss` | Current loss value | Increasing |
| `training_gradient_norm` | Gradient magnitude | >100 or <0.01 |
| `training_convergence_velocity` | Rate of improvement | <0 |
| `training_perplexity` | LLM quality metric | Increasing |
| `training_loss_spike_detected` | Anomaly detection | =1 |

---

## Semantic Alerts

The shift from infrastructure to semantic alerting:

| Traditional | AI-Aware |
|------------|----------|
| "GPU is down" | "Training stalled despite healthy infra" |
| "Memory full" | "Throughput dropped 40%" |
| "High temperature" | "Model stopped converging" |

Configured alerts in `alerts/ai-training-rules.yml`:
- `HiddenIOBottleneck` - GPU high but throughput low
- `TrainingStalledDespiteHealthyInfra` - Cross-layer correlation
- `GradientExplosion` - Norm > 100
- `LossNotDecreasing` - No improvement for 5 minutes

---

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Training        |     |    VMAgent        |     | VictoriaMetrics   |
|   Simulator       |---->|   (scraper)       |---->|    (TSDB)         |
|   :9101           |     |   :8429           |     |    :8428          |
+-------------------+     +-------------------+     +--------+----------+
                                                             |
                          +-------------------+              |
                          |    VMAlert        |<-------------+
                          |  (alerting)       |              |
                          |   :8880           |              |
                          +-------------------+              |
                                                             |
                          +-------------------+              |
                          |    Grafana        |<-------------+
                          | (visualization)   |
                          |   :3000           |
                          +-------------------+
```

**Stack**: VictoriaMetrics + VMAgent + VMAlert + Grafana

**Why VictoriaMetrics?**
- 10x less RAM than Prometheus
- Native high-cardinality support
- Long-term retention
- PromQL compatible
- European (open-source, no vendor lock-in)

---

## File Structure

```
.
├── docker-compose.yml         # Stack definition
├── prometheus.yml             # Scrape configuration
├── Dockerfile.simulator       # Training simulator image
├── scripts/
│   └── training_simulator.py  # Metrics generator (all 3 layers)
├── alerts/
│   └── ai-training-rules.yml  # VMAlert rules
├── grafana/
│   └── provisioning/
│       ├── datasources/       # VictoriaMetrics config
│       └── dashboards/
│           └── json/          # Pre-built dashboards
├── www/
│   └── index.html             # Landing page
├── demo.sh                    # Helper script
└── README.md
```

---

## Real-World Integration

For production deployments:

1. **GPU Metrics**: DCGM Exporter → Prometheus format
2. **Training Metrics**: PyTorch/TensorFlow callbacks → custom exporter
3. **Correlation**: Slurm job IDs as labels across all layers
4. **Alerting**: VMAlert with cross-layer rules

See: [AI Observability Hub](https://github.com/erythix/ai-observability-hub)

---

## Key Takeaways

1. **Utilization is not efficiency** - High GPU% can mask critical bottlenecks
2. **Three layers are essential** - Infrastructure → Workload → Model Health
3. **Cross-layer correlation is the game changer** - Identify root cause in minutes
4. **Open-source = sovereignty** - Full stack, no vendor lock-in

---

**Contact**  
Samuel Desseaux | [erythix.com](https://erythix.com) | VictoriaMetrics Training Partner

*"moins de bruit, plus de terrain"*
