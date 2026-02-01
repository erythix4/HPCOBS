#!/bin/bash
# AI Observability Demo - Scenario Switcher
# FOSDEM 2026 - Samuel Desseaux

set -e

SCENARIOS=("normal" "io_bottleneck" "gradient_explosion" "loss_plateau")

usage() {
    echo "AI Observability Demo - Scenario Switcher"
    echo ""
    echo "Usage: $0 <scenario>"
    echo ""
    echo "Available scenarios:"
    echo "  normal             - All metrics healthy (default)"
    echo "  io_bottleneck      - GPU high, throughput drops (hidden I/O problem)"
    echo "  gradient_explosion - Gradients explode, training unstable"
    echo "  loss_plateau       - Model stops improving"
    echo ""
    echo "Example:"
    echo "  $0 io_bottleneck"
    echo ""
    echo "URLs:"
    echo "  Grafana:          http://localhost:3000 (pass: fosdem2026)"
    echo "  VictoriaMetrics:  http://localhost:8428"
    echo "  Metrics:          http://localhost:9101/metrics"
echo "  Landing Page:     http://localhost:8080"
}

if [ $# -eq 0 ]; then
    usage
    exit 0
fi

SCENARIO=$1

# Validate scenario
valid=false
for s in "${SCENARIOS[@]}"; do
    if [ "$s" == "$SCENARIO" ]; then
        valid=true
        break
    fi
done

if [ "$valid" == "false" ]; then
    echo "❌ Invalid scenario: $SCENARIO"
    echo ""
    usage
    exit 1
fi

echo "🔄 Switching to scenario: $SCENARIO"
echo ""

# Stop current stack
echo "⏹️  Stopping current demo..."
docker-compose down 2>/dev/null || true

# Start with new scenario
echo "▶️  Starting with scenario: $SCENARIO"
SCENARIO=$SCENARIO docker-compose up -d

echo ""
echo "✅ Demo started with scenario: $SCENARIO"
echo ""
echo "📊 Open Grafana: http://localhost:3000"
echo "   Password: fosdem2026"
echo ""
echo "⏳ Wait ~30 seconds for metrics to populate..."
echo ""

# Describe what to expect
case $SCENARIO in
    normal)
        echo "📈 Expected behavior:"
        echo "   - GPU utilization: 85-95%"
        echo "   - Throughput: ~1000 samples/sec (stable)"
        echo "   - Loss: Decreasing smoothly"
        echo "   - All alerts: Clear"
        ;;
    io_bottleneck)
        echo "📈 Expected behavior:"
        echo "   - GPU utilization: 88-94% (looks healthy!)"
        echo "   - Throughput: Drops to ~600 samples/sec"
        echo "   - Data loading ratio: Rises to 45%"
        echo "   - Key insight: GPU high but throughput low = I/O bottleneck"
        ;;
    gradient_explosion)
        echo "📈 Expected behavior:"
        echo "   - GPU utilization: Normal"
        echo "   - Gradient norm: Explodes >100"
        echo "   - Loss: Increases instead of decreasing"
        echo "   - Alert: GradientExplosion fires"
        ;;
    loss_plateau)
        echo "📈 Expected behavior:"
        echo "   - GPU utilization: ~87%"
        echo "   - Loss: Stagnates at ~1.15"
        echo "   - Gradient norm: Near zero"
        echo "   - Alert: LossNotDecreasing fires"
        ;;
esac
