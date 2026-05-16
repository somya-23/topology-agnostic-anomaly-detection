#!/bin/bash
# =============================================================================
# O-RAN srsRAN FALCON Setup Launcher
# Starts all containers, traffic generator, and automated data collection.
# Collects NORMAL data for N hours, then ANOMALY data with all faults x2.
#
# Usage:
#   ./start_setup.sh                    # Full automated pipeline
#   ./start_setup.sh --clean            # Tear down first, then full pipeline
#   ./start_setup.sh --normal-hours 3   # Custom normal duration
#   ./start_setup.sh --fault-reps 1     # Single fault repetition
#   ./start_setup.sh --setup-only       # Only start containers, no collection
# =============================================================================

set -e
# Script lives inside experimental_phase0/, so parent is the project root
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# ─── Parse Arguments ──────────────────────────────────────────────────
CLEAN=false
SETUP_ONLY=false
NORMAL_HOURS=1
FAULT_REPS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)        CLEAN=true; shift ;;
        --setup-only)   SETUP_ONLY=true; shift ;;
        --normal-hours) NORMAL_HOURS="$2"; shift 2 ;;
        --fault-reps)   FAULT_REPS="$1"; shift 1 ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── InfluxDB Token (required by scrapper) ───────────────────────────
export INFLUXDB_TOKEN="605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"

# ─── Cleanup trap (Ctrl+C kills child processes) ─────────────────────
cleanup_on_exit() {
    echo ""
    warn "Caught interrupt — cleaning up background processes..."
    pkill -f "test_traffic.py" 2>/dev/null || true
    pkill -f "scrapper_prometheus_backup.py" 2>/dev/null || true
    for c in srsue2 ; do
        docker exec "$c" pkill -9 iperf 2>/dev/null || true
    done
    log "Background processes killed. Containers still running."
    log "To tear down containers too: ./start_setup.sh --clean"
    exit 0
}
trap cleanup_on_exit SIGINT SIGTERM

wait_for_container() {
    local name=$1
    local timeout=${2:-30}
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    err "$name did not start within ${timeout}s"
    return 1
}

wait_for_healthy() {
    local name=$1
    local timeout=${2:-120}
    local elapsed=0
    log "Waiting for $name to become healthy..."
    while [ $elapsed -lt $timeout ]; do
        local status
        status=$(docker inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null)
        if [ "$status" = "healthy" ]; then
            log "$name is healthy"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    err "$name did not become healthy within ${timeout}s"
    return 1
}

check_rrc_release() {
    local ue=$1
    docker logs --tail 100 "$ue" 2>/dev/null | grep -q "Received RRC Release"
    return $?
}

handle_rrc_release() {
    local ue=$1
    warn "$ue received RRC Release — restarting..."
    docker restart "$ue"
    sleep 15
    if docker ps --format '{{.Names}}' | grep -q "^${ue}$"; then
        log "$ue restarted successfully"
    else
        err "$ue failed to restart!"
        return 1
    fi
}

# ─── Optional clean start ────────────────────────────────────────────
if [ "$CLEAN" = true ]; then
    log "Tearing down all containers..."
    docker compose -f "${BASE_DIR}/docker-compose-ue++.yaml" down 2>/dev/null || true
    docker compose -f "${BASE_DIR}/docker-compose-du.yaml"   down 2>/dev/null || true
    docker compose -f "${BASE_DIR}/docker-compose-cu++.yaml" down 2>/dev/null || true
    # Tear down influxdb/grafana/metrics_server from root compose
    docker compose -f "${BASE_DIR}/docker-compose.yaml" down 2>/dev/null || true
    docker stop prometheus cadvisor 2>/dev/null || true
    docker rm   prometheus cadvisor 2>/dev/null || true
    pkill -f "scrapper_prometheus_backup.py" 2>/dev/null || true
    pkill -f "test_traffic.py" 2>/dev/null || true
    for c in srsue2 open5gs_5gc; do
        docker exec "$c" pkill -9 iperf 2>/dev/null || true
    done
    log "Cleanup done"
    sleep 2
fi

# ─── Step 0: Ensure Docker networks exist ────────────────────────────
log "=== Step 0: Docker Networks ==="
# oran-intel: one subnet covering 175.53.x.x (CU/DU IPs)
docker network create oran-intel --subnet=175.53.0.0/16 2>/dev/null \
    && log "Created oran-intel" || log "oran-intel already exists"
# telemetry-net: separate network for monitoring stack
docker network create telemetry-net --subnet=171.53.0.0/16 2>/dev/null \
    && log "Created telemetry-net" || log "telemetry-net already exists"

# # ─── Step 1: Monitoring stack (influxdb, grafana, metrics_server) ────
# log "=== Step 1: Starting Monitoring Stack (influxdb + grafana + metrics_server) ==="
# # These are defined in root docker-compose.yaml
# docker compose -f "${BASE_DIR}/docker-compose.yaml" up -d influxdb grafana metrics_server redis
# sleep 5

# # Verify influxdb is up before anything else writes to it
# if ! docker ps --format '{{.Names}}' | grep -q '^influxdb$'; then
#     err "influxdb failed to start. Check: docker compose -f docker-compose.yaml logs influxdb"
#     exit 1
# fi
# log "influxdb is running"

# ─── Step 2: Prometheus + cAdvisor ───────────────────────────────────
log "=== Step 2: Starting Prometheus + cAdvisor ==="

if ! docker ps --format '{{.Names}}' | grep -q '^cadvisor$'; then
    docker run -d \
        --name cadvisor \
        --network oran-intel \
        -p 8080:8080 \
        -v /:/rootfs:ro \
        -v /var/run:/var/run:rw \
        -v /sys:/sys:ro \
        -v /var/lib/docker/:/var/lib/docker:ro \
        gcr.io/cadvisor/cadvisor:latest
    log "cAdvisor started"
else
    log "cAdvisor already running"
fi

if ! docker ps --format '{{.Names}}' | grep -q '^prometheus$'; then
    docker run -d \
        --name prometheus \
        --network oran-intel \
        -p 9090:9090 \
        -v "${BASE_DIR}/setup/prometheus:/prometheus-data" \
        prom/prometheus:latest \
        --config.file=/prometheus-data/prometheus.yml
    log "Prometheus started"
else
    log "Prometheus already running"
fi

sleep 3

# ─── Step 3: Core Network (5GC) + CUs ────────────────────────────────
log "=== Step 3: Starting 5GC + CUs (docker-compose-cu++.yaml) ==="
docker compose -f "${BASE_DIR}/docker-compose-cu++.yaml" up -d 5gc

wait_for_healthy open5gs_5gc 120
docker compose -f "${BASE_DIR}/docker-compose-cu++.yaml" up -d cu1


# for cu in srscu0 srscu1 srscu2; do
wait_for_container srscu1 30
# done
log "CU1 started"
sleep 5

# ─── Step 4: DUs ─────────────────────────────────────────────────────
log "=== Step 4: Starting DUs (docker-compose-du.yaml) ==="
docker compose -f "${BASE_DIR}/docker-compose-du.yaml" up -d du2

# for du in srsdu0 srsdu1 srsdu2 srsdu3 srsdu4 srsdu5; do
wait_for_container srsdu2 30
# done
log "DU2 started"

log "Waiting 15s for F1 connections to establish..."
sleep 15

# ─── Step 5: UEs ─────────────────────────────────────────────────────
log "=== Step 5: Starting UEs (docker-compose-ue++.yaml) ==="
docker compose -f "${BASE_DIR}/docker-compose-ue++.yaml" up -d ue2
docker compose -f "${BASE_DIR}/docker-compose-ue++.yaml" up -d grafana influxdb metrics-server

for ue in srsue2; do
    wait_for_container srsue2 30
done
log "UE2 started"

log "Waiting 30s for UE attachment and PDU sessions..."
sleep 30

# ─── Step 6: RRC Release Check & Verification ────────────────────────
log "=== Step 6: Verification + RRC Release Check ==="

echo ""
log "Container Status:"
docker ps --format 'table {{.Names}}\t{{.Status}}' \
    | grep -E 'srs|open5gs|cadvisor|prometheus|influxdb|grafana|metrics|redis' \
    | sort

echo ""
log "Checking UE PDU sessions (tun_srsue interfaces)..."
ATTACHED=0
for ue in srsue2; do
    ip=$(docker exec "$ue" ip -o addr show tun_srsue 2>/dev/null | awk '{print $4}' || true)
    if [ -n "$ip" ]; then
        log "  $ue: $ip"
        ATTACHED=$((ATTACHED + 1))
    else
        warn "  $ue: NO PDU session"
    fi
done

echo ""
log "Checking for RRC Release in UE logs..."
RRC_OK=true
for ue in srsue2; do
    if check_rrc_release "$ue"; then
        warn "  $ue: RRC Release DETECTED — restarting..."
        handle_rrc_release "$ue"
        RRC_OK=false
    else
        log "  $ue: No RRC Release [OK]"
    fi
done

if [ "$RRC_OK" = false ]; then
    warn "Some UEs were restarted. Waiting 45s for re-attachment..."
    sleep 45
    ATTACHED=0
    for ue in srsue2; do
        ip=$(docker exec "$ue" ip -o addr show tun_srsue 2>/dev/null | awk '{print $4}' || true)
        [ -n "$ip" ] && ATTACHED=$((ATTACHED + 1))
    done
    log "After RRC recovery: $ATTACHED/1 UEs attached"
fi

echo ""
log "Checking interface assignments (eth0 should be 175.x)..."
IFACE_OK=true
for c in srscu1 srsdu2; do
    eth0_ip=$(docker exec "$c" ip -o addr show eth0 2>/dev/null \
        | grep -oP '175\.\d+\.\d+\.\d+' | head -1 || true)
    if [ -n "$eth0_ip" ]; then
        log "  $c eth0 = $eth0_ip [OK]"
    else
        warn "  $c eth0 does NOT have 175.x — entrypoint swap may have failed"
        IFACE_OK=false
    fi
done

echo ""
echo "=============================================="
log "Setup complete!"
log "  Containers: $(docker ps --format '{{.Names}}' | wc -l) running"
log "  UEs attached: $ATTACHED/1"
if [ "$IFACE_OK" = true ]; then
    log "  Interfaces: All correct"
else
    warn "  Interfaces: Some containers have swapped eth0/eth1"
fi

# ─── Stop here if --setup-only ───────────────────────────────────────
if [ "$SETUP_ONLY" = true ]; then
    echo ""
    log "Setup-only mode. Exiting without starting data collection."
    log "To start manually:"
    log "  export INFLUXDB_TOKEN=${INFLUXDB_TOKEN}"
    log "  python3 ${BASE_DIR}/experimental_phase0/test_traffic.py &"
    log "  python3 ${BASE_DIR}/experimental_phase0/scrapper_prometheus_backup.py \\"
    log "      --normal-hours ${NORMAL_HOURS} --fault-reps ${FAULT_REPS}"
    echo "=============================================="
    exit 0
fi

# ─── Step 7: Ensure output directory is writable ─────────────────────
DATA_DIR="${BASE_DIR}/experimental_phase0/data_fs_plane"
mkdir -p "$DATA_DIR"
chmod -R 755 "$DATA_DIR" 2>/dev/null || true

# ─── Step 8: Start Traffic Generator ─────────────────────────────────
log "=== Step 8: Starting Traffic Generator ==="
TRAFFIC_LOG="${DATA_DIR}/traffic_gen.log"

pkill -f "test_traffic.py" 2>/dev/null || true
log "Killing leftover iperf sessions inside containers..."
for c in srsue2 open5gs_5gc; do
    docker exec "$c" pkill -9 iperf 2>/dev/null || true
done
sleep 1

# traffic generator is in experimental_phase0/ (not trafficGenerator/)
nohup python3 "${BASE_DIR}/experimental_phase0/test_traffic.py" \
    > "$TRAFFIC_LOG" 2>&1 &
TRAFFIC_PID=$!
log "Traffic generator started (PID: $TRAFFIC_PID)"
log "Traffic log: $TRAFFIC_LOG"

# ─── Step 9: Wait for Metrics to Stabilize ───────────────────────────
log "=== Step 9: Waiting 2 minutes for metrics to stabilize ==="
sleep 120

# ─── Step 10: Start Automated Data Collection ─────────────────────────
log "=== Step 10: Starting Automated Data Collection Pipeline ==="
SCRAPPER_LOG="${DATA_DIR}/scrapper_pipeline.log"

pkill -f "scrapper_prometheus_backup.py" 2>/dev/null || true
sleep 1

# INFLUXDB_TOKEN is already exported above — scrapper will inherit it
nohup python3 "${BASE_DIR}/experimental_phase0/scrapper_prometheus_backup.py" \
    --normal-hours "$NORMAL_HOURS" \
    --fault-reps 0 \
    --anomaly-hours 1 \
    > "$SCRAPPER_LOG" 2>&1 &
SCRAPPER_PID=$!

# Give scrapper 5s to confirm it didn't crash immediately
sleep 5
if ! kill -0 "$SCRAPPER_PID" 2>/dev/null; then
    err "Scrapper died immediately. Check log:"
    tail -20 "$SCRAPPER_LOG"
    kill "$TRAFFIC_PID" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=============================================="
info "AUTOMATED PIPELINE RUNNING"
echo "=============================================="
log "  Traffic Generator PID : $TRAFFIC_PID"
log "  Scrapper Pipeline PID : $SCRAPPER_PID"
echo ""
log "  NORMAL phase      : ${NORMAL_HOURS}h baseline collection (all UEs, full rate)"
log "  HALF-TRAFFIC phase: 1h srsue2 only at 50% rate (no stress injection)"
log "  Est. total        : ~2h"
echo ""
log "  Scrapper log  : tail -f $SCRAPPER_LOG"
log "  Traffic log   : tail -f $TRAFFIC_LOG"
log "  Normal CSV    : ${BASE_DIR}/experimental_phase0/traffic_testing/train_normal.csv"
log "  Anomaly CSV   : ${BASE_DIR}/experimental_phase0/traffic_testing/test_normal_half.csv"
echo ""
log "  Stop all      : kill $TRAFFIC_PID $SCRAPPER_PID"
echo "=============================================="