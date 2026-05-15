# srsRAN with RIC, grafana and Prometheus $Setup Guide

## Prerequisites
Ensure your system is up to date before proceeding with the installation.

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Install cmake and make
sudo apt-get update
sudo apt-get install -y cmake make

```

## Docker and dependencies Installation

The following script removes any existing Docker installations and installs the latest version along with srsRAN_4G dependencies:

```bash
#!/bin/bash
$
# Remove existing Docker packages
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    sudo apt-get remove -y $pkg;
done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository:
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and other dependencies for srsUE_4G
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin build-essential cmake libfftw3-dev libmbedtls-dev libboost-program-options-dev libconfig++-dev libsctp-dev libzmq3-dev


# Verify Docker installation
sudo docker run hello-world

# Add user to Docker group
sudo groupadd docker
sudo usermod -aG docker $USER

# Enable Docker services
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

To apply the group changes, either reboot or run:
```bash
newgrp docker
```
Then, verify Docker runs without sudo:
```bash
docker run hello-world
```
Then, create the docker networks required for the setup:
```bash
docker network create --subnet=175.0.0.0/8 oran-intel
docker network create --subnet=171.0.0.0/8 telemetry-net
```

## Cloning and Setting Up srsRAN

### Step 1: Clone the srsRAN Project Repository and checkout
```bash
cd RAN
git clone https://github.com/srsran/srsRAN_Project.git
cd srsRAN_Project
git checkout d5fa4f0ecb
cd ../..
```

### Step 2: Build UE
```bash
cd UE/
git clone https://github.com/srsran/srsRAN_4G.git
cd srsRAN_4G
mkdir build
cd build
cmake ../
make -j$(nproc)
cd ../../../
```

### Step 3: Clone and Set Up RIC
```bash
mkdir -p RIC
cd RIC/
git clone https://github.com/srsran/oran-sc-ric.git
cd ../
```

### Step 4: Python dependencies
```bash
pip3 install aiohttp influxdb_client aiocsv psutil
```

## Running the Setup

### Step 5: Copy configuration files
```bash
cp -f setup/srsRAN_Project/docker-compose.yml RAN/srsRAN_Project/docker/docker-compose.yml
cp -f setup/srsRAN_Project/open5gs.env RAN/srsRAN_Project/docker/open5gs/open5gs.env
cp -f setup/srsRAN_Project/subscriber_db.csv RAN/srsRAN_Project/docker/open5gs/subscriber_db.csv
cp -f setup/oran-sc-ric/docker-compose.yml RIC/oran-sc-ric/docker-compose.yml
cp -f setup/srsRAN_Project/Dockerfile RAN/srsRAN_Project/docker/Dockerfile
cp -f setup/srsRAN_Project/install_dependencies.sh RAN/srsRAN_Project/docker/scripts/install_dependencies.sh
cp -f setup/srsRAN_Project/cu.cpp RAN/srsRAN_Project/apps/cu/cu.cpp
```

### Step 6: Start the Multi-CU/DU/UE Setup

The setup uses three separate compose files. **CUs must be started sequentially** (5s gap) due to AMF SCTP connection handling.

#### Terminal 1: Start Metrics Stack (metrics-server, influxdb, grafana) + UEs
```bash
docker compose -f docker-compose-ue++.yaml up -d metrics-server influxdb grafana redis
```
Wait for influxdb to be ready (~10s), then:

#### Terminal 2: Start 5GC + CUs (sequentially)
```bash
docker compose -f docker-compose-cu++.yaml up -d 5gc
sleep 10
docker compose -f docker-compose-cu++.yaml up -d cu0
sleep 5
docker compose -f docker-compose-cu++.yaml up -d cu1
sleep 5
docker compose -f docker-compose-cu++.yaml up -d cu2
```

Verify all CUs connected to AMF:
```bash
docker logs srscu0 2>&1 | grep "CU started"
docker logs srscu1 2>&1 | grep "CU started"
docker logs srscu2 2>&1 | grep "CU started"
```

#### Terminal 3: Start DUs
```bash
docker compose -f docker-compose-du.yaml up -d
```

#### Terminal 4: Start UEs
```bash
docker compose -f docker-compose-ue++.yaml up -d ue0 ue1 ue2 ue3 ue4 ue5
```

### Step 7: Verify the Setup

#### Check all containers are running
```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "srs|open5gs|metrics|influx|grafana"
```

#### Verify network interface assignment
```bash
# eth0 should be oran-intel (175.x.x.x), eth1 should be telemetry-net (171.x.x.x)
docker exec srsdu0 ip -o addr show eth0   # should show 175.53.1.10
docker exec srsdu0 ip -o addr show eth1   # should show 171.53.1.10
```

#### Verify metrics pipeline
```bash
# Check DU can reach metrics-server
docker exec srsdu0 ping -c 2 171.40.1.4

# Check metrics-server logs
docker logs metrics_server 2>&1 | tail -5

# Check Grafana at http://localhost:3300
```

### Stopping the Setup

```bash
docker compose -f docker-compose-ue++.yaml down
docker compose -f docker-compose-du.yaml down
docker compose -f docker-compose-cu++.yaml down
```

### Starting from Scratch (full reset)

```bash
docker compose -f docker-compose-ue++.yaml down
docker compose -f docker-compose-du.yaml down
docker compose -f docker-compose-cu++.yaml down
docker network rm oran-intel telemetry-net
docker network create --subnet=175.0.0.0/8 oran-intel
docker network create --subnet=171.0.0.0/8 telemetry-net
```

Then follow Step 6 again.

### Optional: Start containerStateChecker
```bash
nohup python3 containerStateChecker/monitor.py --resource-constraint yes > /dev/null 2>&1 &
```

## Dual-Network Architecture

The setup uses two isolated Docker networks to separate data plane traffic from metrics collection:

### Networks

| Network | Subnet | Purpose |
|---------|--------|---------|
| `oran-intel` | `175.0.0.0/8` | Data plane: F1AP, F1U, N2/N3, ZMQ RF |
| `telemetry-net` | `171.0.0.0/8` | Metrics: DU/CU → metrics-server → InfluxDB → Grafana |

### Why Separate Networks?

Separating metrics from data plane ensures that:
1. **tc packet loss experiments** on `eth0` (oran-intel) affect only srsRAN data traffic, not metric collection
2. Metrics are collected reliably even during network impairment experiments
3. Clear traffic isolation between control/user plane and observability

### Interface Assignment (eth0 / eth1)

Dual-homed containers (DUs, CUs) connect to both networks. Docker assigns interfaces **alphabetically by network name**:

- `oran-intel` (o) → **eth0** (data plane)
- `telemetry-net` (t) → **eth1** (metrics)

This is why the metrics network is named `telemetry-net` (not `metrics-net`) — to ensure `oran-intel` gets `eth0`. This matters because:
- `tc` experiments target `eth0` to affect srsRAN data plane traffic
- All existing scripts and configs assume `eth0` is the primary data interface

### IP Address Assignment

| Container | oran-intel (eth0) | telemetry-net (eth1) | Role |
|-----------|-------------------|-------------------|------|
| 5GC | 175.53.1.2 | - | 5G Core |
| cu0 | 175.53.10.1 + .1.11* | 171.53.10.1 | CU |
| cu1 | 175.53.10.2 + .1.12* | 171.53.10.2 | CU |
| cu2 | 175.53.10.3 + .1.15* | 171.53.10.3 | CU |
| du0 | 175.53.1.10 | 171.53.1.10 | DU → cu0 |
| du1 | 175.53.1.13 | 171.53.1.13 | DU → cu0 |
| du2 | 175.53.1.14 | 171.53.1.14 | DU → cu1 |
| du3 | 175.53.1.16 | 171.53.1.16 | DU → cu2 |
| du4 | 175.53.1.17 | 171.53.1.17 | DU → cu2 |
| du5 | 175.53.1.18 | 171.53.1.18 | DU → cu2 |
| ue0-ue5 | 175.53.2-7.1 | - | UEs |
| metrics-server | - | 171.40.1.4 | Metric collector |
| influxdb | - | 171.40.1.5 | Time-series DB |
| grafana | - | 171.40.1.6 | Dashboard |
| redis | 175.24.0.1 | - | Cache |

\* = secondary IP added via `entrypoint.sh` (dynamic interface detection)

### Entrypoint Scripts

CU entrypoint scripts (`RAN/entrypoint_*.sh`) dynamically detect the oran-intel interface instead of hardcoding `eth0`:

```bash
IFACE=$(ip -o addr show | grep "175\." | awk '{print $2}' | head -1)
ip addr add 175.53.1.11/16 dev ${IFACE}
```

### tc Packet Loss Behavior & the HARQ Experiment

#### What `tc netem loss` Does

The experiment (`experimental_phase0/harq_packet_loss_experiment.py`) runs:

```bash
docker exec srsdu0 tc qdisc replace dev eth0 root netem loss X%
```

**Critical detail:** `tc netem` only affects **EGRESS** (outgoing) packets on srsdu0's eth0. It does NOT affect ingress. This means:

```
AFFECTED (DU egress on eth0):
  DU --[F1U UDP]--> CU     : UL user data forwarded to CU — packets DROPPED (no retransmit, UDP)
  DU --[F1AP SCTP]--> CU   : control plane — SCTP retransmits (resilient)
  DU --[ZMQ TCP]--> UE     : DL IQ samples — TCP retransmits (adds latency)

NOT AFFECTED (DU ingress):
  CU --[F1U UDP]--> DU     : DL user data from CU — arrives fine
  UE --[ZMQ TCP]--> DU     : UL IQ samples from UE — arrives fine
```

#### Why UL Bitrate Increases (or Doesn't Drop) — The Layer Mismatch

This is the key insight: **the scheduler metrics measure the wrong layer for this experiment.**

```
Layer diagram with tc loss on DU egress:

     UE                     DU (srsdu0)                    CU
      |                       |                             |
      |---ZMQ TCP (UL IQ)--->|  PHY decodes PUSCH          |
      |   (ingress, NO loss) |  CRC passes ✓               |
      |                       |  ul_brate counted here ✓    |
      |                       |                             |
      |                       |---F1U UDP (UL data)-------->|
      |                       |   (egress, tc DROPS some)   |
      |                       |                         [LOST HERE]
      |                       |                             |
      |                       |                        App throughput
      |                       |                        drops here
```

The `ul_brate` metric is computed at the **DU scheduler** from CRC results:

```cpp
// scheduler_metrics_handler.cpp
// ul_brate = sum of CRC-passed bytes / report_period
// This counts what PHY decoded, NOT what reached CU
```

Since ZMQ uses TCP (lossless), UL IQ samples from UE always arrive at DU perfectly. PHY always decodes PUSCH successfully. So `ul_brate` stays high or even increases because:

1. UE sends PUSCH data → DU receives via ZMQ TCP (ingress, unaffected) → CRC always passes ✓
2. DU forwards to CU via F1U UDP (egress, tc drops some) → **but scheduler doesn't know about this**
3. `ul_brate` counts step 1 (PHY goodput), not step 2 (F1U delivery)

The `ul_brate` may actually **increase** because:
- tc drops some DL ZMQ TCP segments (DU→UE egress) → TCP retransmits → added latency
- UE misses some DL PDCCH/PDSCH → less DL activity
- Scheduler may allocate slightly more UL resources in response

#### Where the Actual Throughput Drop Happens

The application-level (iperf) throughput drops because:

```
UL iperf data path:
  UE app → UE PDCP → UE RLC → UE MAC → PUSCH → [ZMQ TCP] → DU PHY → DU RLC → [F1U UDP] → CU
                                                                                    ↑
                                                                              tc drops here
                                                                                    ↓
                                                                          CU PDCP doesn't receive
                                                                          all data → iperf sees
                                                                          reduced throughput
```

The loss is at **F1U transport layer** (between DU and CU), not at the **radio/PHY layer** (between UE and DU).

#### Which Metrics Actually Show the Anomaly

**Metrics that WON'T change (PHY-layer, measured before F1U):**

| Metric | Why unchanged |
|--------|---------------|
| `ul_nof_ok` / `ul_nof_nok` | CRC pass/fail at PHY. ZMQ is lossless, so CRC always passes. |
| `ul_brate` | PHY-level goodput. Counts CRC-passed bytes before F1U forwarding. |
| `ul_mcs` | Based on PUSCH SNR which is perfect in ZMQ. |
| HARQ retransmission counts | HARQ operates at PHY layer. No PHY errors in ZMQ → no HARQ retransmissions. |

**Metrics that MAY change (timing/scheduling effects from ZMQ DL latency):**

| Metric | Why it might change |
|--------|---------------------|
| `late_dl_harqs` / `late_ul_harqs` | ZMQ TCP retransmits on DL (DU→UE egress) add latency. If IQ samples arrive late, DU may miss slot deadlines. |
| `avg_crc_delay` / `max_crc_delay` | Processing delays may increase if ZMQ latency disrupts slot timing. |
| `nof_pucch_f0f1_invalid_harqs` | If DL ZMQ latency causes UE to miss PDSCH, UE can't send HARQ ACK → DTX detected. |
| `dl_nof_nok` | If DL ZMQ latency causes UE to miss PDSCH decoding → UE NACKs. |
| `dl_brate` | May drop if DL ZMQ disruption prevents UE from receiving PDSCH. |

**Metrics NOT in InfluxDB (but would show F1U loss):**

| Metric | Layer | Why not available |
|--------|-------|-------------------|
| RLC retransmission count | RLC (DU) | DU sends RLC metrics via JSON, but metrics-server only parses `ue_list`, not RLC reports |
| PDCP discard/reordering | PDCP (CU) | CU metrics not collected to InfluxDB in this setup |
| F1U packet loss counters | F1U transport | Not exposed as a metric |
| iperf throughput | Application | Not collected in InfluxDB — only in iperf stdout |

**The real indicator of the anomaly** is the gap between:
- `ul_brate` (high — PHY sees no loss) vs iperf output (dropping — F1U loses packets)

This gap proves that F1U transport loss is occurring even though the radio layer is healthy.

#### Metrics on `eth1` (telemetry-net) — Unaffected

Metrics collection is on `eth1` (telemetry-net, 171.x.x.x), completely isolated from tc on eth0. Data collection remains reliable throughout the experiment.

#### Summary: tc Loss in ZMQ Setup

| Layer | Transport | tc Effect | Observable? |
|-------|-----------|-----------|-------------|
| PHY (UE↔DU) | ZMQ TCP | TCP retransmits, adds latency | Partially — timing metrics may shift |
| F1U (DU→CU) | UDP | **Packets dropped permanently** | NOT in current InfluxDB metrics |
| F1AP (DU↔CU) | SCTP | SCTP retransmits (resilient) | Minimal impact |
| Application | End-to-end | **Throughput drops** | Only in iperf output, not InfluxDB |

**HARQ retransmissions will NOT trigger** because HARQ operates at the PHY/MAC layer, and ZMQ provides a perfect lossless radio link. To observe HARQ, you would need a real RF channel or a PHY-level error injection (not tc on the transport network).

### Architecture Diagram

See `docs/O-RAN_Network_Architecture.pptx` for the full visual architecture (regenerate with `python3 docs/generate_architecture_ppt.py`).

## Metrics Pipeline: From 5G Radio to InfluxDB

This section documents the complete metrics collection pipeline — how srsRAN scheduler metrics are collected, serialized, transported, and stored in InfluxDB for monitoring via Grafana.

### 5G NR Background

#### Key 5G NR Channels

| Channel | Direction | Purpose |
|---------|-----------|---------|
| **PDSCH** | DL (gNB->UE) | Carries user data downlink |
| **PUSCH** | UL (UE->gNB) | Carries user data uplink |
| **PUCCH** | UL (UE->gNB) | Carries control info: HARQ ACK/NACK, SR, CSI |
| **PDCCH** | DL (gNB->UE) | Carries scheduling grants (DCI) |
| **PRACH** | UL (UE->gNB) | Random access preamble |

#### Key 5G NR Concepts

| Concept | Meaning |
|---------|---------|
| **HARQ** | Hybrid Automatic Repeat Request — retransmission mechanism |
| **CQI** | Channel Quality Indicator (0-15) — UE reports DL channel quality |
| **MCS** | Modulation and Coding Scheme — higher = more throughput, less robust |
| **RI** | Rank Indicator — number of MIMO layers |
| **SNR** | Signal-to-Noise Ratio (dB) |
| **RSRP** | Reference Signal Received Power (dB) |
| **BSR** | Buffer Status Report — UE tells gNB how much UL data it has |
| **PHR** | Power Headroom Report — how much TX power the UE has left |
| **TA** | Timing Advance — compensates propagation delay |
| **CRC** | Cyclic Redundancy Check — verifies UL data integrity |
| **K1** | Slots between PDSCH and PUCCH HARQ feedback |
| **K2** | Slots between DCI and PUSCH transmission |
| **RNTI** | Radio Network Temporary Identifier — per-cell UE address assigned dynamically during connection setup |
| **PCI** | Physical Cell Identity — unique per cell/DU, configured statically |
| **DCI** | Downlink Control Information — scheduling grant carried on PDCCH |
| **Grant** | Permission from gNB to use specific radio resources (time, frequency, MCS) |
| **SR** | Scheduling Request — UE asks gNB for an UL grant |

#### What is a Grant?

A grant is the gNB telling the UE: "You are allowed to use these specific radio resources, right now." Without a grant, the UE **cannot transmit** on PUSCH (only PUCCH/PRACH have pre-configured resources). Everything in 5G NR is **scheduled by the gNB**.

```
+-------------------------------------+
|          UL GRANT (DCI)             |
|                                     |
|  WHO:    UE with RNTI 0x4601       |
|  WHEN:   Slot 7 (K2 slots from now)|
|  WHERE:  PRBs 10-25 (frequency)    |
|  HOW:    MCS 15 (16QAM, rate 0.6)  |
|  SIZE:   1200 bytes max            |
|  HARQ:   Process ID 3             |
+-------------------------------------+
```

- **DL grant** = gNB allocates PDSCH resources to send data **to** UE
- **UL grant** = gNB allocates PUSCH resources for UE to send data **to** gNB

Grants are carried on **PDCCH** as **DCI**. The UE monitors PDCCH every slot looking for DCIs addressed to its RNTI. This is why `nof_failed_pdcch_allocs` is a metric — if the gNB can't fit a DCI on PDCCH, no one gets scheduled.

#### UL Scheduling: How gNB Knows What to Grant

The gNB doesn't guess — the UE tells it via SR and BSR:

```
UE has UL data
    |
    v
1. UE sends Scheduling Request (SR) on PUCCH
   "Hey gNB, I have data to send"
    |
    v
2. gNB grants a small UL allocation on PDCCH
   "Here's a small PUSCH grant"
    |
    v
3. UE sends Buffer Status Report (BSR) as MAC CE on PUSCH
   "I have exactly 54,321 bytes queued across these logical channels"
    |
    v
4. gNB scheduler now knows HOW MUCH data UE has
   Uses BSR + PUSCH SNR + UE power headroom (PHR) to decide:
   - How many PRBs to allocate
   - Which MCS to use
   - How often to schedule this UE
    |
    v
5. gNB sends properly-sized UL grants (DCI on PDCCH)
   UE sends data on PUSCH in the granted resources
```

The metrics that reflect this UL scheduling loop:

| Metric | Role in UL scheduling |
|--------|----------------------|
| `bsr` | BSR value — tells gNB how much UL data UE has queued. If `bsr > 0`, UE still has pending data. |
| `pusch_snr_db` | gNB measures UL channel quality from received PUSCH. Higher SNR = can use higher MCS = more throughput per grant. |
| `last_phr` | UE's power headroom. If low, UE is near max TX power — gNB should use lower MCS or fewer PRBs to avoid errors. |
| `ul_mcs` | The MCS gNB chose based on SNR + PHR. Higher = more bits per resource block. |
| `ul_nof_ok` / `ul_nof_nok` | CRC pass/fail — feedback loop. If `nof_nok` rises, gNB's OLLA (Outer Loop Link Adaptation) lowers MCS. |

Full loop: **SR -> small grant -> BSR -> sized grants -> PUSCH data -> CRC check -> adapt MCS**.

#### DL HARQ Flow

```
Slot N:    gNB sends PDSCH (data)       -->  UE receives
Slot N+K1: UE sends HARQ ACK/NACK on PUCCH  -->  gNB
           If NACK -> gNB retransmits on PDSCH
```

#### UL HARQ Flow

```
Slot N:    gNB sends UL grant (DCI on PDCCH)  -->  UE
Slot N+K2: UE sends data on PUSCH             -->  gNB checks CRC
           If CRC fails -> gNB sends new UL grant for retransmission
```

### Pipeline Architecture

```
+----------------------------------------------------------------------+
|                        gNB (C++ srsRAN)                              |
|                                                                      |
|  +--------------+    +----------------------+    +-----------------+ |
|  |  PHY Layer   |--->| Scheduler            |--->| Metrics Handler | |
|  |  (CRC, UCI)  |    | (ue_event_manager)   |    | (scheduler_     | |
|  +--------------+    +----------------------+    |  metrics_handler | |
|                                                  |  .cpp)           | |
|  Events:                                         +--------+--------+ |
|  - CRC indication (UL)                                    |          |
|  - UCI indication (DL HARQ ACK/NACK)          Every N ms  |          |
|  - BSR, PHR, SR                               (report     |          |
|  - Scheduling results (grants)                 period)     |          |
|                                                           v          |
|                                          +------------------------+  |
|                                          | JSON Consumer          |  |
|                                          | (scheduler_metrics_    |  |
|                                          |  consumers.cpp)        |  |
|                                          +-----------+------------+  |
|                                                      |               |
|                                          +-----------v------------+  |
|                                          | srslog UDP Sink        |  |
|                                          | (fetch_udp_sink)       |  |
|                                          +-----------+------------+  |
+-----------------------------------------|------------+---------------+
                                          |            |
                                          | telemetry  | UDP JSON
                                          | -net       | port 55555
                                          |            v
+----------------------------------------------------------------------+
|  metrics_server (Python)  -->  InfluxDB  -->  Grafana                |
|  (__main__.py)                                                       |
|                                                                      |
|  Parses JSON -> writes to InfluxDB measurement "ue_info"             |
|  Tags: pci, rnti, testbed                                           |
|  Fields: everything else from ue_container                           |
+----------------------------------------------------------------------+
```

### Pipeline Stages

#### Stage 1: Event Collection (C++ — scheduler_metrics_handler.cpp)

Raw events arrive from the scheduler and are accumulated in `non_persistent_data` per UE:

**DL Events:**

| Handler | Trigger | What it accumulates |
|---------|---------|---------------------|
| `handle_slot_result()` | Every slot | `dl_mcs`, `nof_dl_cws`, `tot_dl_prbs_used` from PDSCH grants |
| `handle_dl_harq_ack()` | UCI HARQ-ACK decoded | `count_uci_harq_acks` (if ACK), `count_uci_harqs` (always), `sum_dl_tb_bytes` (if ACK) |
| `handle_uci_with_harq_ack()` | HARQ feedback received | `sum_pucch_harq_delay_slots` or `sum_pusch_harq_delay_slots` |
| `handle_harq_timeout()` | No HARQ response | `count_uci_harqs` (for DL, counted as NACK) |
| `handle_uci_pdu_indication()` | UCI PDU from PHY | `nof_pucch_f0f1_invalid_harqs`, `nof_pucch_f2f3f4_invalid_harqs`, CSI/CQI/RI |

**UL Events:**

| Handler | Trigger | What it accumulates |
|---------|---------|---------------------|
| `handle_slot_result()` | Every slot | `ul_mcs`, `nof_puschs`, `tot_ul_prbs_used` from PUSCH grants |
| `handle_crc_indication()` | PHY CRC result | `count_crc_acks` (if pass), `count_crc_pdus` (always), `sum_ul_tb_bytes` (if pass), SNR, RSRP, TA |
| `handle_harq_timeout()` | No CRC response | `count_crc_pdus` (for UL, counted as fail) |
| `handle_ul_bsr_indication()` | UE BSR report | `last_bsr` |
| `handle_ul_phr_indication()` | UE PHR report | `last_phr` |

#### Stage 2: Report Computation (compute_report())

Every `report_period` ms (configured as `du_report_period` in du config), accumulated data is converted into `scheduler_ue_metrics`:

```cpp
// DL metrics
ret.dl_mcs        = avg of accumulated dl_mcs
ret.dl_brate_kbps = sum_dl_tb_bytes * 8 / report_period   // only ACKed bytes (goodput)
ret.dl_nof_ok     = count_uci_harq_acks                    // HARQ ACKs
ret.dl_nof_nok    = count_uci_harqs - count_uci_harq_acks  // HARQ NACKs + timeouts

// UL metrics
ret.ul_mcs        = avg of accumulated ul_mcs
ret.ul_brate_kbps = sum_ul_tb_bytes * 8 / report_period   // only CRC-passed bytes (goodput)
ret.ul_nof_ok     = count_crc_acks                          // CRC passes
ret.ul_nof_nok    = count_crc_pdus - count_crc_acks         // CRC fails + timeouts
```

Then all counters are reset to zero for the next period.

#### Stage 3: JSON Serialization (scheduler_metrics_consumers.cpp)

The JSON consumer serializes the report using `DECLARE_METRIC` macros. The first argument is the JSON field name:

```cpp
DECLARE_METRIC("ul_nof_ok", metric_ul_nof_ok, unsigned, "");
//              ^ JSON key   ^ C++ name        ^ type
```

Resulting JSON sent over UDP:

```json
{
  "timestamp": 1711234567.890,
  "ue_list": [
    {
      "ue_container": {
        "pci": 1,
        "rnti": 17921,
        "cqi": 12,
        "dl_mcs": 27,
        "dl_brate": 50000000.0,
        "dl_nof_ok": 150,
        "dl_nof_nok": 2,
        "ul_mcs": 25,
        "ul_brate": 20000000.0,
        "ul_nof_ok": 80,
        "ul_nof_nok": 1,
        "bsr": 0,
        "pusch_snr_db": 25.3,
        "avg_pucch_harq_delay": 3.0
      }
    }
  ]
}
```

#### Stage 4: InfluxDB Storage (metrics_server/__main__.py)

The Python metrics server receives JSON over UDP, extracts `pci` and `rnti` as InfluxDB tags, and pushes all remaining fields as InfluxDB fields:

```python
ue_container = ue_info["ue_container"]
rnti = ue_container.pop("rnti")     # becomes TAG
pci = ue_container.pop("pci")       # becomes TAG
# everything else becomes FIELDS automatically
```

**InfluxDB measurement:** `ue_info`
**Tags:** `pci`, `rnti`, `testbed`
**Fields:** all remaining keys from `ue_container`

### All InfluxDB Metrics Reference

#### DL Metrics (sent to InfluxDB under measurement `ue_info`)

| InfluxDB Field | Source | 5G Meaning |
|----------------|--------|------------|
| `cqi` | CSI report from UE on PUCCH | Channel Quality Indicator (0-15). UE measures DL reference signals and reports quality. Higher = better DL channel. gNB uses this to select MCS. |
| `dl_ri` | CSI report from UE | DL Rank Indicator. Number of usable MIMO spatial layers for DL. RI=2 means 2x throughput vs RI=1. |
| `dl_mcs` | Scheduler grant decision | Modulation and Coding Scheme (0-28). Higher MCS = higher spectral efficiency but needs better channel. 0-9: QPSK, 10-16: 16QAM, 17-28: 64QAM/256QAM. |
| `dl_brate` | Sum of ACKed DL TB bytes | DL bitrate in bps. Only counts bytes for which HARQ ACK was received (goodput, not raw throughput). |
| `dl_nof_ok` | HARQ ACKs received | Count of successfully received DL transport blocks. Each ACK means UE decoded PDSCH correctly. |
| `dl_nof_nok` | HARQ NACKs + timeouts | Count of failed DL transport blocks. Each NACK = UE couldn't decode, triggers DL retransmission. |
| `dl_bs` | DL buffer state in gNB | Pending DL data (bytes) waiting to be sent to this UE. High value = gNB has data but can't send fast enough. |

#### UL Metrics

| InfluxDB Field | Source | 5G Meaning |
|----------------|--------|------------|
| `ul_mcs` | Scheduler grant decision | UL MCS chosen by gNB based on UL channel quality (PUSCH SNR). |
| `ul_brate` | Sum of CRC-passed UL TB bytes | UL bitrate in bps. Only counts bytes where CRC passed (goodput). |
| `ul_nof_ok` | CRC pass on PUSCH | Count of successfully decoded UL transport blocks. gNB runs CRC check on received PUSCH data. |
| `ul_nof_nok` | CRC fail + timeouts | Count of failed UL transport blocks. CRC fail = data corrupted, triggers UL retransmission grant. |
| `ul_ri` | SRS measurement at gNB | UL Rank Indicator from SRS. Number of usable UL MIMO layers. |
| `bsr` | BSR MAC CE from UE | Buffer Status Report. UE tells gNB how many bytes of UL data it has queued. gNB uses this to allocate PUSCH resources. |
| `last_phr` | PHR MAC CE from UE | Power Headroom Report (dB). How much TX power margin UE has. Low PHR = UE near max power, may need lower MCS. |
| `pusch_snr_db` | PHY measurement at gNB | UL signal quality on PUSCH. gNB measures SNR of received PUSCH. Used for UL link adaptation (MCS selection). |
| `pusch_rsrp_db` | PHY measurement at gNB | UL signal power on PUSCH. Indicates path loss. |
| `pucch_snr_db` | PHY measurement at gNB | UL signal quality on PUCCH. Affects reliability of HARQ ACK/NACK, SR, CSI feedback. |

#### Timing Metrics

| InfluxDB Field | Source | 5G Meaning |
|----------------|--------|------------|
| `ta_ns` | TA from CRC/PUCCH/SRS | Timing Advance in nanoseconds. Distance-proportional (~3.3ns per meter). Used to align UE transmissions at gNB. |
| `pusch_ta_ns` | TA from CRC indication | TA measured from PUSCH. |
| `pucch_ta_ns` | TA from PUCCH | TA measured from PUCCH. |
| `srs_ta_ns` | TA from SRS | TA measured from Sounding Reference Signal (most accurate). |

#### HARQ Delay Metrics

| InfluxDB Field | Source | Meaning |
|----------------|--------|---------|
| `avg_crc_delay` | `last_slot_tx - sl_rx` for CRC | Avg processing delay (ms) from PUSCH reception to scheduler. |
| `max_crc_delay` | Max of above | Worst-case UL processing delay. |
| `avg_pucch_harq_delay` | `last_slot_tx - sl_rx` for PUCCH | Avg processing delay (ms) from PUCCH HARQ feedback to scheduler. Constant ~3ms is normal (pipeline latency). |
| `max_pucch_harq_delay` | Max of above | Worst-case PUCCH processing delay. |
| `avg_pusch_harq_delay` | `last_slot_tx - sl_rx` for PUSCH UCI | Avg processing delay (ms) from PUSCH-carried HARQ feedback to scheduler. |
| `max_pusch_harq_delay` | Max of above | Worst-case PUSCH UCI processing delay. |
| `avg_ce_delay` | Control element processing | Avg MAC CE processing delay. |
| `max_ce_delay` | Max of above | Worst-case MAC CE delay. |

#### Invalid HARQ / DTX Metrics

| InfluxDB Field | Source | Meaning |
|----------------|--------|---------|
| `nof_pucch_f0f1_invalid_harqs` | DTX on PUCCH F0/F1 | UE was expected to send HARQ ACK/NACK on PUCCH Format 0/1, but nothing detected (DTX). Indicates UE didn't transmit or severe channel. |
| `nof_pucch_f2f3f4_invalid_harqs` | DTX on PUCCH F2/F3/F4 | Same but on higher PUCCH formats (used for multiple HARQ bits or CSI). |
| `nof_pusch_invalid_harqs` | DTX on PUSCH UCI | HARQ feedback expected multiplexed on PUSCH, but DTX detected. |
| `nof_pucch_f2f3f4_invalid_csis` | Invalid CSI on PUCCH | CSI report received but flagged invalid. |
| `nof_pusch_invalid_csis` | Invalid CSI on PUSCH | CSI on PUSCH flagged invalid. |

### RNTI and PCI: How UEs Are Identified in Metrics

#### RNTI Is Assigned Dynamically, Not Configured

RNTI (Radio Network Temporary Identifier) is **not set in config files**. It is assigned dynamically by each DU's MAC layer when a UE connects via random access:

```
UE sends PRACH preamble
    |
    v
DU MAC assigns a temporary RA-RNTI
    |
    v
After RRC setup completes, DU MAC assigns a C-RNTI
(e.g., 0x4601, 0x4602, etc.)
```

Each DU/cell has its **own independent RNTI space**. When only 1 UE connects to each cell, the MAC will likely assign the same first RNTI (e.g., `0x4601`) to all of them. This is normal — RNTI is only unique **within a cell**.

#### PCI Is Configured Per DU (Static)

PCI (Physical Cell Identity) is set in each DU config file:

| DU | Config File | PCI |
|----|------------|-----|
| du0 | `RAN/du_zmq.conf` | 1 |
| du1 | `RAN/du_zmq_1.conf` | 2 |
| du2 | `RAN/du_zmq_2.conf` | 3 |
| du3 | `RAN/du_zmq_3.conf` | 4 |
| du4 | `RAN/du_zmq_4.conf` | 5 |
| du5 | `RAN/du_zmq_5.conf` | 6 |

#### Why Same RNTI Across Cells Is Not a Problem

InfluxDB uses **all tags combined** as the unique key. The metrics-server stores UE metrics with both `pci` and `rnti` as tags:

```python
"tags": {
    "pci": pci,          # different per DU (1, 2, 3, 4, 5, 6)
    "rnti": f"{rnti:x}", # might be same (4601) across cells
    "testbed": testbed,
}
```

So even if RNTI is the same, each UE is uniquely identified:

| pci | rnti | InfluxDB tag set | Unique? |
|-----|------|------------------|---------|
| 1 | 4601 | `pci=1,rnti=4601` | Yes |
| 2 | 4601 | `pci=2,rnti=4601` | Yes |
| 3 | 4601 | `pci=3,rnti=4601` | Yes |
| 4 | 4601 | `pci=4,rnti=4601` | Yes |
| 5 | 4601 | `pci=5,rnti=4601` | Yes |
| 6 | 4601 | `pci=6,rnti=4601` | Yes |

**No data mixing occurs.** Each UE's metrics are correctly separated.

#### Grafana Queries: Always Group by PCI

When querying InfluxDB in Grafana, always filter or group by `pci` to distinguish UEs:

```flux
from(bucket: "srsran")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "ue_info")
  |> filter(fn: (r) => r["_field"] == "ul_brate")
  |> group(columns: ["pci", "rnti"])   // group by BOTH to separate UEs
```

### DU Metrics Configuration

Metrics are configured in the DU config file (e.g., `RAN/du_zmq.conf`):

```yaml
metrics:
  addr: 171.40.1.4                # metrics-server IP on telemetry-net
  port: 55555                     # UDP port
  enable_json: true               # enable JSON serialization
  layers:
    enable_app_usage: true
    enable_sched: true            # scheduler metrics (the ones in this doc)
    enable_rlc: true
    enable_mac: true
    enable_executor: true
    enable_du_low: true
    enable_ru: true
  periodicity:
    app_usage_report_period: 500  # ms
    du_report_period: 500         # ms - how often metrics are computed and sent
```

### How to Add a Custom Metric

This section walks through adding a new metric end-to-end. Example: adding `ul_nof_retx` and `dl_nof_retx` to track actual HARQ retransmission counts.

#### Files to Modify

| Step | File | Change |
|------|------|--------|
| 1 | `lib/scheduler/logging/scheduler_metrics_handler.h` | Add counter to `non_persistent_data` struct |
| 2 | `include/srsran/scheduler/scheduler_metrics.h` | Add field to `scheduler_ue_metrics` struct |
| 3 | `lib/scheduler/logging/scheduler_metrics_handler.cpp` | Collect data in event handler + compute in `compute_report()` |
| 4 | `apps/.../consumers/scheduler_metrics_consumers.cpp` | `DECLARE_METRIC`, add to metric set, write in JSON consumer |
| 5 | metrics-server Python | **No changes needed** (generic field handling) |
| 6 | InfluxDB | **No changes needed** (schema-less, new fields appear automatically) |

All file paths are relative to `RAN/srsRAN_Project/`.

#### Step 1: Add Counters (scheduler_metrics_handler.h)

In the `non_persistent_data` struct, add new counters:

```cpp
struct non_persistent_data {
  // ... existing fields ...
  unsigned count_dl_harq_retxs = 0;  // DL HARQ retransmissions
  unsigned count_ul_harq_retxs = 0;  // UL HARQ retransmissions
};
```

#### Step 2: Add Report Fields (scheduler_metrics.h)

In the `scheduler_ue_metrics` struct, add output fields:

```cpp
struct scheduler_ue_metrics {
  // ... existing fields ...
  unsigned dl_nof_retx = 0;
  unsigned ul_nof_retx = 0;
};
```

#### Step 3: Collect and Compute (scheduler_metrics_handler.cpp)

**Collect** — in `handle_slot_result()`, inside the grants loops:

```cpp
// Inside DL grants loop (~line 444):
for (const dl_msg_alloc& dl_grant : slot_result.dl.ue_grants) {
    // ... existing code ...
    // ADD: count DL retransmissions (new_data=false means retx)
    if (!dl_grant.pdsch_cfg.codewords[0].new_data) {
      ues[it->second].data.count_dl_harq_retxs++;
    }
}

// Inside UL grants loop (~line 466):
for (const ul_sched_info& ul_grant : slot_result.ul.puschs) {
    // ... existing code ...
    // ADD: count UL retransmissions (new_data=false means retx)
    if (!ul_grant.pusch_cfg.new_data) {
      ues[it->second].data.count_ul_harq_retxs++;
    }
}
```

**Compute** — in `compute_report()`:

```cpp
ret.dl_nof_retx = data.count_dl_harq_retxs;
ret.ul_nof_retx = data.count_ul_harq_retxs;
```

No reset changes needed — `data = {}` already zeroes the entire struct.

#### Step 4: JSON Serialization (scheduler_metrics_consumers.cpp)

**Declare the metrics** (after existing DECLARE_METRIC lines):

```cpp
DECLARE_METRIC("dl_nof_retx", metric_dl_nof_retx, unsigned, "");
DECLARE_METRIC("ul_nof_retx", metric_ul_nof_retx, unsigned, "");
```

**Add to the metric set** (inside `DECLARE_METRIC_SET("ue_container", ...)`):

```cpp
DECLARE_METRIC_SET("ue_container",
                   mset_ue_container,
                   // ... existing metrics ...
                   metric_dl_nof_retx,
                   metric_ul_nof_retx,
                   // ... rest of existing metrics ...
);
```

**Write in the JSON consumer** (inside `handle_metric()`, after `ul_nof_nok`):

```cpp
output.write<metric_dl_nof_ok>(ue.dl_nof_ok);
output.write<metric_dl_nof_nok>(ue.dl_nof_nok);
output.write<metric_dl_nof_retx>(ue.dl_nof_retx);   // NEW
output.write<metric_ul_nof_ok>(ue.ul_nof_ok);
output.write<metric_ul_nof_nok>(ue.ul_nof_nok);
output.write<metric_ul_nof_retx>(ue.ul_nof_retx);   // NEW
```

#### Step 5: Rebuild and Deploy

**No changes needed to docker-compose files, Dockerfile, metrics-server, or InfluxDB.**

The metrics-server Python code generically pushes all `ue_container` fields:
```python
"fields": dict(convert_integers_to_floats(ue_container).items())
```

Only rebuild the Docker image and restart:

```bash
# Rebuild the srsRAN image (used by both CU and DU)
docker compose -f docker-compose-cu++.yaml build cu0

# Restart DUs (where scheduler metrics are collected)
docker compose -f docker-compose-du.yaml down
docker compose -f docker-compose-du.yaml up -d

# Restart CUs if needed
docker compose -f docker-compose-cu++.yaml down
docker compose -f docker-compose-cu++.yaml up -d
```

#### Step 6: Query in InfluxDB

New fields appear automatically in InfluxDB. Query them with Flux:

```flux
from(bucket: "srsran")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "ue_info")
  |> filter(fn: (r) => r["_field"] == "ul_nof_retx" or r["_field"] == "dl_nof_retx")
  |> group(columns: ["rnti"])
```

Or compute HARQ retransmission rate:

```flux
nok = from(bucket: "srsran")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "ue_info")
  |> filter(fn: (r) => r["_field"] == "ul_nof_nok")

ok = from(bucket: "srsran")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "ue_info")
  |> filter(fn: (r) => r["_field"] == "ul_nof_ok")

// BLER = nok / (ok + nok) gives the block error rate
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `RAN/srsRAN_Project/lib/scheduler/logging/scheduler_metrics_handler.h` | Metric counter structs and handler class |
| `RAN/srsRAN_Project/lib/scheduler/logging/scheduler_metrics_handler.cpp` | Event collection, accumulation, and `compute_report()` |
| `RAN/srsRAN_Project/include/srsran/scheduler/scheduler_metrics.h` | `scheduler_ue_metrics` report struct |
| `RAN/srsRAN_Project/apps/.../consumers/scheduler_metrics_consumers.cpp` | JSON/stdout/log serialization and `DECLARE_METRIC` macros |
| `RAN/srsRAN_Project/lib/scheduler/ue_scheduling/ue_event_manager.cpp` | Routes PHY events (CRC, UCI, HARQ) to metrics handler |
| `RAN/srsRAN_Project/docker/metrics_server/src/metrics_server/__main__.py` | Python UDP receiver, writes to InfluxDB |
| `RAN/du_zmq.conf` | DU config with metrics address/port/periodicity |
| `docker-compose-ue++.yaml` | metrics-server, influxdb, grafana services |

