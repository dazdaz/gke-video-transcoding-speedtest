# README.md

# FFmpeg GPU Parallel Transcoding on GKE

A comprehensive solution for GPU-accelerated video transcoding on Google Kubernetes Engine (GKE) using NVIDIA NVENC hardware encoding.

This was wrote, in order to dive into how video transcoding on GKE with MPS enabled actually works, and to explore what configuration achieves the fastest video transcoding using ffmpeg.

## Overview

This project provides tools and scripts to orchestrate parallel video transcoding jobs on GKE using NVIDIA GPUs. It's optimized for production workloads with support for Google Cloud Storage integration, hardware-accelerated encoding, and comprehensive performance monitoring.

## Architecture

```
┌─────────────────────────────────────────────┐
│       Python Control Script                 │
│       (03-transcode-test.py)                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│       Kubernetes Job YAML Generation        │
│    • Dynamic configuration                  │
│    • GPU resource requests                  │
│    • Node pool selection                    │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│       Kubernetes Job Deployment             │
│    • Parallel pod creation                  │
│    • GPU scheduling                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│          Pod Execution                      │
│    ┌──────────────────────────────┐         │
│    │  FFmpeg Parallel Streams     │         │
│    │  • NVENC hardware encoding   │         │
│    │  • GPU monitoring            │         │
│    │  • Metrics collection        │         │
│    └──────────────────────────────┘         │
└─────────────────────────────────────────────┘
```

## Key Features

### Multi-Mode Operation
- **Default Mode**: Generates test videos using FFmpeg's lavfi source
- **GCS Mode**: Reads from and writes to Google Cloud Storage buckets
- **Local Mode**: Processes local video files (auto-uploads large files to GCS)

### GPU Optimization
- Hardware-accelerated encoding using NVIDIA NVENC
- Support for MPS (Multi-Process Service) for GPU sharing
- Configurable parallel streams per GPU
- Automatic GPU utilization monitoring

### Kubernetes Integration
- Dynamic job YAML generation
- Node pool selection for GPU nodes
- Resource limits and requests management
- Automatic job cleanup
- Workload Identity support

### Comprehensive Monitoring
- Real-time GPU utilization tracking
- NVENC encoder utilization metrics
- Per-stream performance statistics
- Final aggregated results

## Prerequisites

### System Requirements
- GKE cluster with GPU nodes (NVIDIA T4, L4, A10, or similar)
- `kubectl` configured to access the cluster
- Python 3.8+
- `gsutil` (for GCS operations)

### Kubernetes Requirements
- GPU drivers installed on nodes
- NVIDIA device plugin deployed
- Optional: MPS daemon for GPU sharing

### Permissions
- Kubernetes RBAC permissions for job creation
- GCS read/write permissions (for GCS mode)
- Workload Identity configured (recommended)

## Quick Start

### 1. Build and Deploy Container

```bash
# Build the FFmpeg container with NVENC support
./01-build-container.sh

# Configure IAM permissions
./02-iam-roles.sh
```

### 2. Run a Simple Test

```bash
# Install Python dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run a basic test with NVENC encoding
./03-transcode-test.py --encoder-type nvenc --gpu-model T4 --num-replicas 1
```

### 3. Process Videos from GCS

```bash
# Process all MP4 files from a GCS bucket
./03-transcode-test.py \
  --mode gcs \
  --encoder-type nvenc \
  --gpu-model T4 \
  --num-replicas 2 \
  --gcs-input-bucket transcode-preprocessing-bucket \
  --gcs-output-bucket transcode-postprocessing-bucket
```

## Usage Examples

### CPU Encoding Test
```bash
./03-transcode-test.py \
  --encoder-type cpu \
  --num-replicas 4
```

### GPU Encoding with Custom Preset
```bash
./03-transcode-test.py \
  --encoder-type nvenc \
  --gpu-model L4 \
  --num-replicas 2 \
  --preset p7 \
  --rc-mode vbr
```

### No-Wait Mode (Fire and Forget)
```bash
./03-transcode-test.py \
  --encoder-type nvenc \
  --gpu-model T4 \
  --num-replicas 1 \
  --no-wait
```

### Manual Cleanup
```bash
./03-transcode-test.py \
  --encoder-type nvenc \
  --gpu-model T4 \
  --num-replicas 1 \
  --no-auto-cleanup
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Input/output mode (default/gcs/local) | default |
| `--num-replicas` | Number of parallel pods | 1 |
| `--streams-per-gpu` | Parallel FFmpeg streams per pod | Auto-detected |
| `--encoder-type` | Encoder type (cpu/nvenc) | cpu |
| `--gpu-model` | GPU model for stream estimation | T4 |
| `--preset` | NVENC preset (p1-p7) | p4 |
| `--rc-mode` | Rate control mode (cbr/vbr/cq) | vbr |
| `--gcs-input-bucket` | GCS input bucket | transcode-preprocessing-bucket |
| `--gcs-output-bucket` | GCS output bucket | transcode-postprocessing-bucket |
| `--no-wait` | Don't wait for job completion | False |
| `--no-auto-cleanup` | Don't auto-cleanup after completion | False |

## NVENC Stream Estimator

The project includes a dedicated tool for estimating optimal stream counts:

```bash
# Estimate streams for T4 GPU with H.264 encoding
./nvenc_stream_estimator.py T4 H.264 p3 CBR 30 --nvenc_units 1

# Estimate for L4 with HEVC
./nvenc_stream_estimator.py L4 HEVC p1 VBR 60 --nvenc_units 2

# Estimate for H100
./nvenc_stream_estimator.py H100 H.264 p4 VBR 30 --nvenc_units 3
```

### Supported GPU Models

| Model | Architecture | NVENC Units | Typical Streams (1080p) |
|-------|--------------|-------------|------------------------|
| T4 | Turing | 1 | 8-10 |
| L4 | Ada Lovelace | 2 | 16-20 |
| A10 | Ampere | 1 | 10-12 |
| A100 | Ampere | 5 | 40-50 |
| H100 | Hopper | 3 | 30-35 |

## Performance Monitoring

### Real-Time Metrics

During execution, the script monitors:
- GPU utilization (%)
- NVENC encoder utilization (%)
- GPU memory usage
- Processing speed (megapixels/second)
- Real-time factor (how many times faster than real-time)

### Sample Output

```
=== PERFORMANCE ANALYSIS ===
✓ EXCELLENT: NVENC is performing well with 8.5x real-time speed
  This confirms hardware acceleration is active and effective

Average Megapixels/sec: 245.32 MP/s
Average Real-time Factor: 8.5x
Successfully transcoded: 10/10 files
```

## Troubleshooting

### Common Issues

**Node Pool Not Found**
```bash
# Check available nodes
kubectl get nodes -l cloud.google.com/gke-nodepool=transcoder-gpunode-pool-mps
```

**Pod Scheduling Failures**
```bash
# Verify GPU resources
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Low NVENC Utilization**
```bash
# Monitor GPU in real-time
kubectl exec -it <pod-name> -- nvidia-smi -l 1
```

**GCS Permission Denied**
```bash
# Check service account
kubectl get serviceaccount gke-tp-service-account -o yaml

# Verify IAM bindings
gcloud projects get-iam-policy daev-playground --flatten="bindings[].members" --filter="bindings.members:gke-tp-service-account"
```

### Debug Commands

```bash
# View job status
kubectl get jobs -l app=ffmpeg-gpu-parallel-nvenc

# Check pod logs
kubectl logs -l app=ffmpeg-gpu-parallel-nvenc --tail=100

# Monitor GPU usage
kubectl exec -it <pod-name> -- nvidia-smi nvenc -s

# Check NVENC sessions
kubectl exec -it <pod-name> -- nvidia-smi nvenc --query
```

## Advanced Configuration

### Custom Video Parameters

Edit the constants in `03-transcode-test.py`:

```python
VIDEO_DURATION = 60              # seconds
VIDEO_RESOLUTION = "3840x2160"   # 4K
VIDEO_FPS = 30
VIDEO_BITRATE = "20M"            # 20 Mbps
```

### Resource Limits

Modify the resources in the YAML generation function:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "16Gi"
    cpu: "8"
  requests:
    nvidia.com/gpu: 1
    memory: "8Gi"
    cpu: "4"
```

### NVENC Encoder Settings

Common presets and their use cases:

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| p1 | Fastest | Lower | Real-time streaming |
| p3 | Fast | Good | Live encoding |
| p4 | Medium | Better | Balanced |
| p7 | Slow | Best | Archival/VOD |

## Best Practices

1. **Start Small**: Begin with minimal configuration and scale up
2. **Monitor Metrics**: Aim for >80% NVENC utilization
3. **Use Workload Identity**: For secure GCS access
4. **Clean Up Resources**: Always cleanup jobs to avoid costs
5. **Test Presets**: Find the right balance for your use case
6. **Regional Colocation**: Use GCS buckets in same region as GKE

## Project Structure

```
.
├── 01-build-container.sh          # Build and push container
├── 02-iam-roles.sh                # Configure IAM permissions
├── 03-transcode-test.py           # Main orchestration script
├── nvenc_stream_estimator.py     # Stream capacity estimator
├── Dockerfile                     # FFmpeg + NVENC container
├── entrypoint.sh                  # Container entrypoint
├── cloudbuild.yaml                # Cloud Build configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided as-is for educational and testing purposes.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the example usage in `example-usage.txt`
- Examine logs with `kubectl logs`
- Monitor GPU with `nvidia-smi`

## Acknowledgments

- NVIDIA for NVENC technology
- FFmpeg project
- Google Cloud Platform
- Kubernetes community
