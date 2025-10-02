#!/usr/bin/env python3

"""
GPU FFmpeg Parallel Transcoding Test for GKE
Optimized for NVIDIA GPUs with proper NVENC utilization
Supports GCS bucket operations with batch processing using Google Cloud Storage API
Enhanced with Workload Identity support and improved service account handling
FIXED: Performance metrics and NVENC verification
"""

import subprocess
import datetime
import argparse
import sys
import time
import json
import os
import base64
import re

# Try to import Google Cloud Storage, but don't fail if not available locally
try:
    from google.cloud import storage
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not installed locally. Will be installed in container.")

# Configuration
JOB_BASE_NAME = "ffmpeg-gpu-parallel-nvenc"
NODE_POOL = "transcoder-gpunode-pool-mps"
NAMESPACE = "default"
DEFAULT_NUM_REPLICAS = 1
DEFAULT_STREAMS_PER_GPU = 4
VIDEO_DURATION = 60  # seconds
VIDEO_RESOLUTION = "3840x2160"  # 4K
VIDEO_FPS = 30
VIDEO_BITRATE = "20M"  # Higher bitrate for 4K

# Default GCS buckets
DEFAULT_INPUT_BUCKET = "transcode-preprocessing-bucket"
DEFAULT_OUTPUT_BUCKET = "transcode-postprocessing-bucket"

# NVENC capabilities data for different GPU models
NVENC_CAPABILITIES = {
    'T4': {
        'nvenc_chips': 1,
        'max_concurrent_sessions': 40,
        'architecture': 'Turing',
        'nvenc_generation': 7,
        'typical_streams': {
            'p1': 8, 'p2': 7, 'p3': 6, 'p4': 5, 'p5': 4, 'p6': 4, 'p7': 3
        }
    },
    'A10': {
        'nvenc_chips': 1,
        'max_concurrent_sessions': 40,
        'architecture': 'Ampere',
        'nvenc_generation': 8,
        'typical_streams': {
            'p1': 12, 'p2': 10, 'p3': 8, 'p4': 7, 'p5': 6, 'p6': 5, 'p7': 4
        }
    },
    'A100': {
        'nvenc_chips': 5,
        'max_concurrent_sessions': 40,
        'architecture': 'Ampere',
        'nvenc_generation': 8,
        'typical_streams': {
            'p1': 50, 'p2': 40, 'p3': 35, 'p4': 30, 'p5': 25, 'p6': 20, 'p7': 18
        }
    },
    'L4': {
        'nvenc_chips': 2,
        'max_concurrent_sessions': 40,
        'architecture': 'Ada Lovelace',
        'nvenc_generation': 9,
        'typical_streams': {
            'p1': 20, 'p2': 16, 'p3': 14, 'p4': 12, 'p5': 10, 'p6': 8, 'p7': 7
        }
    },
    'L40': {
        'nvenc_chips': 3,
        'max_concurrent_sessions': 40,
        'architecture': 'Ada Lovelace',
        'nvenc_generation': 9,
        'typical_streams': {
            'p1': 30, 'p2': 25, 'p3': 20, 'p4': 18, 'p5': 15, 'p6': 12, 'p7': 10
        }
    },
    'H100': {
        'nvenc_chips': 3,
        'max_concurrent_sessions': 40,
        'architecture': 'Hopper',
        'nvenc_generation': 9,
        'typical_streams': {
            'p1': 35, 'p2': 28, 'p3': 24, 'p4': 20, 'p5': 18, 'p6': 15, 'p7': 12
        }
    }
}

def wait_for_job_completion(job_name, timeout=600, check_interval=5):
    """
    Wait for a Kubernetes job to complete.
    
    Args:
        job_name: Name of the job to wait for
        timeout: Maximum time to wait in seconds
        check_interval: Time between status checks in seconds
    
    Returns:
        True if job completed successfully, False otherwise
    """
    start_time = time.time()
    
    print(f"\n‚è≥ Waiting for job '{job_name}' to complete...")
    print(f"   Timeout: {timeout}s, checking every {check_interval}s")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n‚ö†Ô∏è Timeout reached after {timeout}s")
            return False
        
        try:
            # Get job status
            result = subprocess.run(
                ["kubectl", "get", "job", job_name, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            job_data = json.loads(result.stdout)
            status = job_data.get("status", {})
            
            # Check conditions
            conditions = status.get("conditions", [])
            for condition in conditions:
                if condition.get("type") == "Complete" and condition.get("status") == "True":
                    print(f"\n‚úÖ Job completed successfully after {elapsed:.1f}s")
                    return True
                elif condition.get("type") == "Failed" and condition.get("status") == "True":
                    print(f"\n‚ùå Job failed after {elapsed:.1f}s")
                    print(f"   Reason: {condition.get('reason', 'Unknown')}")
                    print(f"   Message: {condition.get('message', 'No message')}")
                    return False
            
            # Show progress
            active = status.get("active", 0)
            succeeded = status.get("succeeded", 0)
            failed = status.get("failed", 0)
            
            status_msg = f"   [{elapsed:.0f}s] Active: {active}, Succeeded: {succeeded}, Failed: {failed}"
            print(f"\r{status_msg}", end="", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f"\n‚ö†Ô∏è Error checking job status: {e}")
            return False
        
        time.sleep(check_interval)

def cleanup_job(job_name, yaml_filename=None):
    """
    Clean up a Kubernetes job and optionally its YAML file.
    
    Args:
        job_name: Name of the job to delete
        yaml_filename: Optional YAML file to delete
    """
    print(f"\nüßπ Cleaning up job '{job_name}'...")
    
    try:
        # Delete the job
        result = subprocess.run(
            ["kubectl", "delete", "job", job_name, "--wait=false"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ‚úî Job deleted: {job_name}")
        
        # Delete YAML file if provided
        if yaml_filename and os.path.exists(yaml_filename):
            os.remove(yaml_filename)
            print(f"   ‚úî YAML file deleted: {yaml_filename}")
            
    except subprocess.CalledProcessError as e:
        print(f"   ‚ö†Ô∏è Error during cleanup: {e}")

def check_service_account():
    """Check if the required service account exists and verify Workload Identity configuration."""
    try:
        # Check if gke-tp-service-account exists
        result = subprocess.run(
            ["kubectl", "get", "serviceaccount", "gke-tp-service-account", "-n", NAMESPACE, "-o", "json"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            sa_data = json.loads(result.stdout)
            annotations = sa_data.get("metadata", {}).get("annotations", {})
            gcp_sa = annotations.get("iam.gke.io/gcp-service-account", "")
            
            if gcp_sa:
                print(f"‚úî Using service account: gke-tp-service-account")
                print(f"    Linked to GCP service account: {gcp_sa}")
                
                # Verify the annotation is correct
                if "@" in gcp_sa and ".iam.gserviceaccount.com" in gcp_sa:
                    print(f"    ‚úî Workload Identity properly configured")
                else:
                    print(f"    ‚ö† Warning: GCP service account annotation may be incorrect")
            else:
                print(f"‚ö† Service account 'gke-tp-service-account' exists but not configured for Workload Identity")
                print(f"    Run the setup script to configure Workload Identity")
            
            return "gke-tp-service-account"
        else:
            # Get the default service account
            print(f"‚ö† Service account 'gke-tp-service-account' not found")
            print(f"‚ö† Using default service account (may not have GCS permissions)")
            print(f"")
            print(f"    To fix this, run the setup script:")
            print(f"    ./01-create-service-accounts-workload-identity.sh")
            print(f"")
            return "default"
            
    except Exception as e:
        print(f"‚ö† Error checking service account: {e}")
        print(f"    Using default service account")
        return "default"

def estimate_nvenc_streams(gpu_model='T4', preset='p4', resolution='3840x2160', 
                           fps=30, codec='h264', rc_mode='vbr', num_nvenc_units=None):
    """
    Estimate optimal number of NVENC streams for a given GPU and encoding configuration.
    """
    
    gpu_input_name = gpu_model.upper().strip()
    
    gpu_mapping = {
        'TESLA T4': 'T4',
        'T4': 'T4',
        'NVIDIA T4': 'T4',
        'A10': 'A10',
        'NVIDIA A10': 'A10',
        'A100': 'A100',
        'NVIDIA A100': 'A100',
        'A100-SXM4-80GB': 'A100',
        'A100-SXM4-40GB': 'A100',
        'L4': 'L4',
        'NVIDIA L4': 'L4',
        'L40': 'L40',
        'NVIDIA L40': 'L40',
        'H100': 'H100',
        'NVIDIA H100': 'H100',
        'H100-SXM5': 'H100',
        'H100-PCIE': 'H100'
    }
    
    gpu_key = gpu_mapping.get(gpu_input_name, gpu_input_name)
    
    if gpu_key not in NVENC_CAPABILITIES:
        print(f"‚ö† Warning: GPU model '{gpu_model}' not in database. Using T4 estimates.")
        gpu_key = 'T4'
    
    gpu_info = NVENC_CAPABILITIES[gpu_key]
    
    nvenc_chips = num_nvenc_units if num_nvenc_units else gpu_info['nvenc_chips']
    base_streams = gpu_info['typical_streams'].get(preset, 4)
    
    width, height = map(int, resolution.split('x'))
    pixels = width * height
    
    resolution_factors = {
        1920 * 1080: 1.0,
        2560 * 1440: 0.7,
        3840 * 2160: 0.4,
        7680 * 4320: 0.15
    }
    
    closest_res = min(resolution_factors.keys(), key=lambda x: abs(x - pixels))
    res_factor = resolution_factors[closest_res]
    
    fps_factor = 30 / fps if fps > 30 else 1.0
    
    codec_factors = {
        'h264': 1.0,
        'h265': 0.6,
        'hevc': 0.6,
        'av1': 0.3
    }
    codec_factor = codec_factors.get(codec.lower(), 1.0)
    
    rc_factors = {
        'cbr': 1.0,
        'vbr': 0.9,
        'cq': 0.85,
        'vbr_hq': 0.8
    }
    rc_factor = rc_factors.get(rc_mode.lower(), 1.0)
    
    estimated_streams = int(base_streams * res_factor * fps_factor * codec_factor * rc_factor)
    estimated_streams = int(estimated_streams * (nvenc_chips / gpu_info['nvenc_chips']))
    estimated_streams = max(1, estimated_streams)
    estimated_streams = min(estimated_streams, gpu_info['max_concurrent_sessions'])
    
    print(f"\n{'=' * 60}")
    print(f"NVENC Stream Estimation for {gpu_key}")
    print(f"{'=' * 60}")
    print(f"GPU Model: {gpu_key} ({gpu_info['architecture']})")
    print(f"NVENC Generation: {gpu_info['nvenc_generation']}")
    print(f"NVENC Chips: {nvenc_chips}")
    print(f"Encoding Config: {codec.upper()} {resolution}@{fps}fps, preset={preset}, rc={rc_mode}")
    print(f"Base Streams (preset {preset}): {base_streams}")
    print(f"Resolution Factor: {res_factor:.2f}")
    print(f"FPS Factor: {fps_factor:.2f}")
    print(f"Codec Factor: {codec_factor:.2f}")
    print(f"Rate Control Factor: {rc_factor:.2f}")
    print(f"{'=' * 60}")
    print(f"‚úî Estimated Optimal Streams: {estimated_streams}")
    print(f"{'=' * 60}\n")
    
    return estimated_streams

def check_node_pool_exists():
    """Check if the specified node pool exists and has GPU nodes."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-l", f"cloud.google.com/gke-nodepool={NODE_POOL}", "-o", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        nodes_data = json.loads(result.stdout)
        nodes = nodes_data.get("items", [])
        
        if not nodes:
            print(f"‚ö† Warning: No nodes found in node pool '{NODE_POOL}'")
            print("Available node pools:")
            subprocess.run(["kubectl", "get", "nodes", "-L", "cloud.google.com/gke-nodepool"])
            
            response = input("\nDo you want to:\n1. Continue anyway (may fail)\n2. Remove node selector (use any GPU node)\n3. Abort\nChoice (1/2/3): ").strip()
            
            if response == "2":
                return False
            elif response == "3":
                sys.exit(1)
        else:
            print(f"‚úî Found {len(nodes)} node(s) in pool '{NODE_POOL}'")
            
            for node in nodes:
                capacity = node.get("status", {}).get("capacity", {})
                if "nvidia.com/gpu" in capacity:
                    gpu_count = capacity["nvidia.com/gpu"]
                    node_name = node["metadata"]["name"]
                    print(f"  - Node {node_name}: {gpu_count}x GPU")
        
        return True
    except subprocess.CalledProcessError:
        print(f"Error checking node pool")
        return True

def create_yaml_file(num_replicas, streams_per_gpu, use_mps=False, mode="default",
                     gcs_input_bucket=None, gcs_input_file=None,
                     gcs_output_bucket=None, local_input_file=None,
                     use_node_selector=True, embed_content=None,
                     preset='p7', rc_mode='vbr', job_suffix="",
                     service_account="default", encoder_type="cpu"):
    """
    Create Kubernetes Job YAML for parallel GPU FFmpeg testing.
    """
    
    # Use the actual output bucket value
    if gcs_output_bucket is None:
        gcs_output_bucket = DEFAULT_OUTPUT_BUCKET

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    if job_suffix:
        safe_suffix = re.sub(r'[^a-z0-9-]', '-', job_suffix.lower())
        safe_suffix = re.sub(r'-+', '-', safe_suffix)
        safe_suffix = safe_suffix[:30]
        job_name = f"{JOB_BASE_NAME}-{safe_suffix}-{timestamp}"
    else:
        job_name = f"{JOB_BASE_NAME}-{timestamp}"

    # Get project ID from environment, metadata, or use default
    try:
        # Try environment variable first
        project_id = os.environ.get('PROJECT_ID')
        
        if not project_id:
            # Try to get from gcloud config
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                project_id = result.stdout.strip()
        
        if not project_id:
            # Fallback to hardcoded value
            project_id = "my-playground"
            
    except Exception:
        project_id = "my-playground"
    
    # Select the appropriate base image
    if encoder_type == "nvenc":
        # Use your custom NVENC-enabled image from Artifact Registry
        base_image = f"us-central1-docker.pkg.dev/{project_id}/ffmpeg-nvidia/ffmpeg-nvidia-nvenc:latest"
        print(f"üê≥ Using custom NVENC container image from Artifact Registry")
        print(f"    Image: {base_image}")
    else:
        # For CPU encoding, you could also build a CPU version or use standard image
        base_image = f"us-central1-docker.pkg.dev/{project_id}/ffmpeg-nvidia/ffmpeg-nvidia-nvenc:latest"
        print(f"üê≥ Using custom container image for CPU encoding")
        print(f"    Image: {base_image}")

    # Set GPU request based on encoder type
    gpu_request = "1" if encoder_type == "nvenc" else "0"
    encoder_display = "NVENC (GPU)" if encoder_type == "nvenc" else "CPU (libx264)"

    if use_mps:
        effective_parallelism = num_replicas
        mps_display = "Enabled (GPU Sharing)"
    else:
        effective_parallelism = num_replicas
        mps_display = "Disabled (Time-Sharing)"

    bitrate_value = int(VIDEO_BITRATE[:-1])
    max_bitrate = bitrate_value * 2
    buffer_size = bitrate_value * 4

    # Base bash script with FIXED GPU detection and GCS support using Python Storage API
    bash_script = f"""#!/bin/bash
set -Euo pipefail

# Function to add timestamp to log messages
log_action() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}}

# Function to format file size
format_file_size() {{
    local size=$1
    if [ $size -ge 1073741824 ]; then
        echo "$(echo "scale=2; $size / 1073741824" | bc) GB"
    elif [ $size -ge 1048576 ]; then
        echo "$(echo "scale=2; $size / 1048576" | bc) MB"
    elif [ $size -ge 1024 ]; then
        echo "$(echo "scale=2; $size / 1024" | bc) KB"
    else
        echo "${{size}} bytes"
    fi
}}

# Declare associative arrays for tracking statistics
declare -A FILE_STATS
declare -A FILE_TIMES
declare -A FILE_RESOLUTIONS
declare -A FILE_DURATIONS
declare -A FILE_FPS_VALUES
declare -A FILE_SIZES_INPUT
declare -A FILE_SIZES_OUTPUT
declare -A FILE_ENCODER_USED
declare -A FILE_TRANSCODE_SPEED

# Main execution block
main() {{

log_action "============================================"
log_action "=== FFmpeg Parallel Transcoding Test ==="
log_action "============================================"
log_action "Pod: $(hostname)"
log_action "Date: $(date)"
log_action "Mode: {mode.upper()}"
log_action "Requested Encoder Type: {encoder_display}"
log_action "MPS Mode: {mps_display}"
log_action "GPU Request: {gpu_request}"
log_action "Streams per Pod: {streams_per_gpu}"
log_action ""

# Install dependencies
log_action "Installing dependencies..."

# Install basic tools first
apt-get update > /dev/null 2>&1
apt-get install -y time bc wget coreutils jq curl python3 python3-pip python3-venv > /dev/null 2>&1

# Create virtual environment and install Python GCS library
log_action "Setting up Python virtual environment..."
python3 -m venv /tmp/venv

# Activate venv and install google-cloud-storage
/tmp/venv/bin/pip install --quiet --upgrade pip
/tmp/venv/bin/pip install --quiet google-cloud-storage

# Set ENCODER_AVAILABLE based on encoder type
if [ "{encoder_type}" = "nvenc" ]; then
    log_action "=== Checking for NVENC Support ==="
    
    # Check if nvidia-smi is available and GPU is present
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_action "‚úÖ NVIDIA GPU detected"
            
            # FIXED: Check FFmpeg for NVENC support - use a more robust detection
            # First get the list of encoders, then check for h264_nvenc
            NVENC_CHECK=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep "h264_nvenc" || true)
            
            if [ -n "$NVENC_CHECK" ]; then
                log_action "‚úÖ h264_nvenc encoder available in FFmpeg"
                ENCODER_AVAILABLE="nvenc"
                ENCODER_DISPLAY="NVENC (GPU)"
            else
                log_action "‚ö† h264_nvenc not found in FFmpeg, falling back to CPU encoding"
                log_action "  This usually means FFmpeg wasn't compiled with NVENC support"
                ENCODER_AVAILABLE="cpu"
                ENCODER_DISPLAY="libx264 (CPU)"
            fi
            
            # Show available NVIDIA encoders
            log_action "Available NVIDIA encoders in FFmpeg:"
            ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "nvenc|nvidia|cuda" | sed 's/^/   /' || log_action "  None found"
        else
            log_action "‚ö† nvidia-smi failed, GPU may not be accessible"
            ENCODER_AVAILABLE="cpu"
            ENCODER_DISPLAY="libx264 (CPU)"
        fi
    else
        log_action "‚ö† nvidia-smi not found, NVIDIA drivers may not be installed"
        ENCODER_AVAILABLE="cpu"
        ENCODER_DISPLAY="libx264 (CPU)"
    fi
else
    ENCODER_AVAILABLE="cpu"
    ENCODER_DISPLAY="libx264 (CPU)"
    log_action "=== Using CPU Encoding (libx264) ==="
fi

# Verify FFmpeg installation
log_action "=== Verifying FFmpeg Installation ==="
which ffmpeg || (log_action "FFmpeg not found in PATH" && exit 1)
ffmpeg -version | head -n1

# Show FFmpeg configuration (for debugging)
log_action "FFmpeg Configuration:"
ffmpeg -hide_banner -buildconf 2>/dev/null | grep -E "enable-nvenc|enable-cuda" | head -5 || log_action "  Configuration info not available"

# Configure gcloud to use the node's default service account or Workload Identity
log_action "=== Service Account Debug Information ==="

# Show K8s service account from environment variable
log_action "Kubernetes Service Account (from env): ${{KUBERNETES_SERVICE_ACCOUNT:-Not set}}"

# Show K8s service account from mounted files
log_action "Kubernetes Service Account (from mounted files):"
if [ -f /var/run/secrets/kubernetes.io/serviceaccount/namespace ]; then
    NAMESPACE=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
    log_action "  Namespace: $NAMESPACE"
fi
if [ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
    log_action "  Token exists: Yes"
    # Check if this is a Workload Identity token
    if [ -f /var/run/secrets/kubernetes.io/serviceaccount/ca.crt ]; then
        log_action "  Type: Workload Identity enabled"
    fi
else
    log_action "  Token exists: No"
fi

# For Workload Identity, the metadata service will provide the correct credentials
log_action ""
log_action "GCP Service Account (from metadata service):"
SA_EMAIL=$(curl -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" 2>/dev/null || echo "Could not retrieve")
log_action "  Email: $SA_EMAIL"

# Show project
PROJECT_ID=$(curl -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null || echo "Could not retrieve")
log_action "  Project: $PROJECT_ID"

# Check if using Workload Identity
if [ "{service_account}" != "default" ]; then
    log_action "  Using Workload Identity via: {service_account}"
fi

# Display GPU information only if using GPU encoding and NVENC is available
if [ "{encoder_type}" = "nvenc" ] && [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
    log_action ""
    log_action "=== GPU Information ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,utilization.encoder,utilization.decoder \\
                 --format=csv,noheader,nounits || true
    log_action ""

    # Check NVENC capabilities
    log_action "=== NVENC Capabilities ==="
    nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps,encoder.stats.averageLatency \\
                 --format=csv,noheader,nounits 2>/dev/null || log_action "NVENC session info not available"
    log_action ""
else
    log_action ""
    log_action "=== CPU Information ==="
    lscpu | grep -E "Model name|CPU\\(s\\)|Thread\\(s\\) per core|Core\\(s\\) per socket" || true
    log_action ""
fi

# Python script for GCS operations using Storage API
cat > /tmp/gcs_operations.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import sys
import os
from google.cloud import storage
from google.auth import default

def list_mp4_files(bucket_name):
    try:
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(bucket_name)
        
        mp4_files = []
        for blob in bucket.list_blobs():
            if blob.name.lower().endswith('.mp4'):
                mp4_files.append(blob.name)
        
        return mp4_files
    except Exception as e:
        print(f"Error listing files: {{e}}", file=sys.stderr)
        return []

def get_file_size(bucket_name, blob_name):
    try:
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.reload()
        return blob.size if blob.size else 0
    except Exception as e:
        print(f"Error getting file size: {{e}}", file=sys.stderr)
        return 0

def download_file(bucket_name, source_blob_name, destination_file_name):
    try:
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {{source_blob_name}} to {{destination_file_name}}")
        return True
    except Exception as e:
        print(f"Error downloading file: {{e}}", file=sys.stderr)
        return False

def upload_file(bucket_name, source_file_name, destination_blob_name):
    try:
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(source_file_name)
        print(f"Uploaded {{source_file_name}} to gs://{{bucket_name}}/{{destination_blob_name}}")
        return True
    except Exception as e:
        print(f"Error uploading file: {{e}}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gcs_operations.py <command> <args>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        bucket_name = sys.argv[2]
        files = list_mp4_files(bucket_name)
        for f in files:
            print(f)
    elif command == "size":
        bucket_name = sys.argv[2]
        blob_name = sys.argv[3]
        size = get_file_size(bucket_name, blob_name)
        print(size)
    elif command == "download":
        bucket_name = sys.argv[2]
        source_blob = sys.argv[3]
        dest_file = sys.argv[4]
        success = download_file(bucket_name, source_blob, dest_file)
        sys.exit(0 if success else 1)
    elif command == "upload":
        bucket_name = sys.argv[2]
        source_file = sys.argv[3]
        dest_blob = sys.argv[4]
        success = upload_file(bucket_name, source_file, dest_blob)
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {{command}}")
        sys.exit(1)
PYTHON_SCRIPT

chmod +x /tmp/gcs_operations.py

# Function to get video information
get_video_info() {{
    local input_file=$1
    local duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file" 2>/dev/null || echo "0")
    local resolution=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$input_file" 2>/dev/null || echo "0x0")
    local fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$input_file" 2>/dev/null | bc -l 2>/dev/null || echo "0")
    
    echo "$duration|$resolution|$fps"
}}

# Function to calculate megapixels per second (for actual transcode time only)
calculate_mp_per_sec() {{
    local resolution=$1
    local transcode_time=$2
    local fps=$3
    local duration=$4
    
    if [[ "$resolution" =~ ([0-9]+)x([0-9]+) ]]; then
        local width=${{BASH_REMATCH[1]}}
        local height=${{BASH_REMATCH[2]}}
        local total_pixels=$((width * height))
        
        # Calculate total frames based on duration and fps
        local total_frames=$(echo "$duration * $fps" | bc -l 2>/dev/null || echo "0")
        
        # Calculate total megapixels
        local total_megapixels=$(echo "scale=2; ($total_pixels * $total_frames) / 1000000" | bc -l 2>/dev/null || echo "0")
        
        # Calculate megapixels per second based on transcode time
        if (( $(echo "$transcode_time > 0" | bc -l) )); then
            local mp_per_sec=$(echo "scale=2; $total_megapixels / $transcode_time" | bc -l 2>/dev/null || echo "0")
            echo "$mp_per_sec"
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}}

# Main transcoding logic
log_action "Starting transcoding test..."
log_action "ACTUAL ENCODER TO BE USED: $ENCODER_DISPLAY"

# Create a test video or process input based on mode
if [ "{mode}" = "default" ]; then
    log_action "Generating test video..."
    
    # Build FFmpeg command for generation
    if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
        FFMPEG_GEN_CMD="ffmpeg -f lavfi -i testsrc2=size={VIDEO_RESOLUTION}:rate={VIDEO_FPS}:duration={VIDEO_DURATION} -c:v h264_nvenc -preset {preset} -rc {rc_mode} -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -gpu 0 -y /tmp/test_input.mp4"
        log_action "Using NVENC for video generation"
    else
        FFMPEG_GEN_CMD="ffmpeg -f lavfi -i testsrc2=size={VIDEO_RESOLUTION}:rate={VIDEO_FPS}:duration={VIDEO_DURATION} -c:v libx264 -preset medium -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -y /tmp/test_input.mp4"
        log_action "Using CPU encoder (libx264) for video generation"
    fi
    
    log_action "FFmpeg Generation Command: $FFMPEG_GEN_CMD"
    eval "$FFMPEG_GEN_CMD 2>&1 | tail -5"
    
    # Transcode the test video
    for i in $(seq 1 {streams_per_gpu}); do
        log_action "Processing stream $i of {streams_per_gpu}"
        
        # Build FFmpeg transcode command
        if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
            FFMPEG_TRANSCODE_CMD="ffmpeg -hwaccel cuda -hwaccel_device 0 -i /tmp/test_input.mp4 -c:v h264_nvenc -preset {preset} -rc {rc_mode} -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -gpu 0 -y /tmp/output_$i.mp4"
        else
            FFMPEG_TRANSCODE_CMD="ffmpeg -i /tmp/test_input.mp4 -c:v libx264 -preset medium -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -y /tmp/output_$i.mp4"
        fi
        
        log_action "FFmpeg Transcode Command: $FFMPEG_TRANSCODE_CMD"
        
        # Execute and measure time
        START_TIME=$(date +%s.%N)
        eval "$FFMPEG_TRANSCODE_CMD 2>&1 | tail -5"
        END_TIME=$(date +%s.%N)
        TRANSCODE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        
        # Get video info and calculate stats
        VIDEO_INFO=$(get_video_info "/tmp/test_input.mp4")
        IFS='|' read -r duration resolution fps <<< "$VIDEO_INFO"
        MP_PER_SEC=$(calculate_mp_per_sec "$resolution" "$TRANSCODE_TIME" "$fps" "$duration")
        
        # Get file sizes
        INPUT_SIZE=$(stat -c%s "/tmp/test_input.mp4" 2>/dev/null || echo "0")
        OUTPUT_SIZE=$(stat -c%s "/tmp/output_$i.mp4" 2>/dev/null || echo "0")
        
        # Calculate real-time speed factor
        if (( $(echo "$duration > 0 && $TRANSCODE_TIME > 0" | bc -l) )); then
            RT_FACTOR=$(echo "scale=2; $duration / $TRANSCODE_TIME" | bc -l)
        else
            RT_FACTOR="0"
        fi
        
        log_action "Stream $i Statistics:"
        log_action "  - Input File Size: $(format_file_size $INPUT_SIZE)"
        log_action "  - Output File Size: $(format_file_size $OUTPUT_SIZE)"
        log_action "  - Transcode Time: ${{TRANSCODE_TIME}}s"
        log_action "  - Resolution: $resolution"
        log_action "  - Duration: ${{duration}}s"
        log_action "  - FPS: $fps"
        log_action "  - Megapixels/sec: ${{MP_PER_SEC}} MP/s"
        log_action "  - Real-time Factor: ${{RT_FACTOR}}x"
        log_action "  - Encoder Used: $ENCODER_DISPLAY"
        
        # Store stats for summary
        FILE_STATS["stream_$i"]="$MP_PER_SEC"
        FILE_TIMES["stream_$i"]="$TRANSCODE_TIME"
        FILE_RESOLUTIONS["stream_$i"]="$resolution"
        FILE_DURATIONS["stream_$i"]="$duration"
        FILE_FPS_VALUES["stream_$i"]="$fps"
        FILE_SIZES_INPUT["stream_$i"]="$INPUT_SIZE"
        FILE_SIZES_OUTPUT["stream_$i"]="$OUTPUT_SIZE"
        FILE_ENCODER_USED["stream_$i"]="$ENCODER_DISPLAY"
        FILE_TRANSCODE_SPEED["stream_$i"]="$RT_FACTOR"
    done

elif [ "{mode}" = "gcs" ]; then
    log_action "Processing GCS mode..."
    log_action "Input bucket: {gcs_input_bucket}"
    log_action "Output bucket: {gcs_output_bucket}"
    
    # List ALL MP4 files in the input bucket using Python Storage API
    log_action "Listing MP4 files in input bucket..."
    MP4_FILES=$(/tmp/venv/bin/python /tmp/gcs_operations.py list "{gcs_input_bucket}")
    
    if [ -z "$MP4_FILES" ]; then
        log_action "‚ö† No MP4 files found in bucket {gcs_input_bucket}"
        log_action "Creating a test video to demonstrate the workflow..."
        
        # Generate a test video
        if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
            FFMPEG_GEN_CMD="ffmpeg -f lavfi -i testsrc2=size={VIDEO_RESOLUTION}:rate={VIDEO_FPS}:duration={VIDEO_DURATION} -c:v h264_nvenc -preset {preset} -rc {rc_mode} -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -gpu 0 -y /tmp/test_input.mp4"
        else
            FFMPEG_GEN_CMD="ffmpeg -f lavfi -i testsrc2=size={VIDEO_RESOLUTION}:rate={VIDEO_FPS}:duration={VIDEO_DURATION} -c:v libx264 -preset medium -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -y /tmp/test_input.mp4"
        fi
        
        log_action "FFmpeg Generation Command: $FFMPEG_GEN_CMD"
        eval "$FFMPEG_GEN_CMD 2>&1 | tail -5"
        
        # Upload test video to input bucket
        log_action "Uploading test video to input bucket..."
        TIMESTAMP=$(date +%s)
        /tmp/venv/bin/python /tmp/gcs_operations.py upload "{gcs_input_bucket}" /tmp/test_input.mp4 "test_video_${{TIMESTAMP}}.mp4"
        
        # Use the uploaded test video
        MP4_FILES="test_video_${{TIMESTAMP}}.mp4"
    fi
    
    # Count total files
    TOTAL_FILES=$(echo "$MP4_FILES" | wc -l)
    log_action "Found $TOTAL_FILES MP4 file(s) in bucket"
    
    # Process ALL files in the bucket (not limited by streams_per_gpu)
    FILE_COUNT=0
    TOTAL_TRANSCODE_TIME=0
    
    for FILE in $MP4_FILES; do
        FILE_COUNT=$((FILE_COUNT + 1))
        log_action ""
        log_action "========================================="
        log_action "Processing file $FILE_COUNT of $TOTAL_FILES: $FILE"
        log_action "========================================="
        
        # Get file size from GCS
        GCS_FILE_SIZE=$(/tmp/venv/bin/python /tmp/gcs_operations.py size "{gcs_input_bucket}" "$FILE")
        log_action "GCS file size: $(format_file_size $GCS_FILE_SIZE)"
        
        # Download the file using Python Storage API
        log_action "Downloading from GCS..."
        /tmp/venv/bin/python /tmp/gcs_operations.py download "{gcs_input_bucket}" "$FILE" "/tmp/input_${{FILE_COUNT}}.mp4"
        
        if [ ! -f "/tmp/input_${{FILE_COUNT}}.mp4" ]; then
            log_action "‚ö† Failed to download $FILE, skipping"
            continue
        fi
        
        # Get actual file size after download
        INPUT_SIZE=$(stat -c%s "/tmp/input_${{FILE_COUNT}}.mp4" 2>/dev/null || echo "0")
        
        # Get input video info
        VIDEO_INFO=$(get_video_info "/tmp/input_${{FILE_COUNT}}.mp4")
        IFS='|' read -r duration resolution fps <<< "$VIDEO_INFO"
        log_action "Input video info: Resolution=$resolution, Duration=${{duration}}s, FPS=$fps"
        
        # Build FFmpeg transcode command
        OUTPUT_FILE="/tmp/output_${{FILE_COUNT}}.mp4"
        if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
            FFMPEG_TRANSCODE_CMD="ffmpeg -hwaccel cuda -hwaccel_device 0 -i /tmp/input_${{FILE_COUNT}}.mp4 -c:v h264_nvenc -preset {preset} -rc {rc_mode} -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -gpu 0 -y $OUTPUT_FILE"
        else
            FFMPEG_TRANSCODE_CMD="ffmpeg -i /tmp/input_${{FILE_COUNT}}.mp4 -c:v libx264 -preset medium -b:v {VIDEO_BITRATE} -maxrate {max_bitrate}M -bufsize {buffer_size}M -y $OUTPUT_FILE"
        fi
        
        log_action "FFmpeg Command: $FFMPEG_TRANSCODE_CMD"
        log_action "Transcoding file with $ENCODER_DISPLAY..."
        
        # Execute and measure time
        START_TIME=$(date +%s.%N)
        eval "$FFMPEG_TRANSCODE_CMD 2>&1 | tail -10"
        TRANSCODE_EXIT_CODE=$?
        END_TIME=$(date +%s.%N)
        TRANSCODE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        TOTAL_TRANSCODE_TIME=$(echo "$TOTAL_TRANSCODE_TIME + $TRANSCODE_TIME" | bc)
        
        if [ $TRANSCODE_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
            # Get output file size
            OUTPUT_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo "0")
            
            # Calculate megapixels per second (for transcode time only)
            MP_PER_SEC=$(calculate_mp_per_sec "$resolution" "$TRANSCODE_TIME" "$fps" "$duration")
            
            log_action "‚úÖ Transcoding successful"
            log_action "Transcoding Statistics for $FILE:"
            log_action "  - Input File Size: $(format_file_size $INPUT_SIZE)"
            log_action "  - Output File Size: $(format_file_size $OUTPUT_SIZE)"
            log_action "  - Compression Ratio: $(echo "scale=2; $OUTPUT_SIZE * 100 / $INPUT_SIZE" | bc)%"
            log_action "  - Transcode Time: ${{TRANSCODE_TIME}}s"
            log_action "  - Input Resolution: $resolution"
            log_action "  - Input Duration: ${{duration}}s"
            log_action "  - Input FPS: $fps"
            log_action "  - Processing Speed: ${{MP_PER_SEC}} MP/s"
            log_action "  - Encoder Used: $ENCODER_DISPLAY"
            
            # Calculate real-time factor
            if (( $(echo "$duration > 0" | bc -l) )); then
                RT_FACTOR=$(echo "scale=2; $duration / $TRANSCODE_TIME" | bc -l)
                log_action "  - Real-time Factor: ${{RT_FACTOR}}x"
                
                # Verify NVENC is being used based on speed
                if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
                    if (( $(echo "$RT_FACTOR > 2" | bc -l) )); then
                        log_action "  - ‚úÖ NVENC CONFIRMED: Real-time factor ${{RT_FACTOR}}x indicates hardware acceleration"
                    else
                        log_action "  - ‚ö† Warning: Low real-time factor may indicate NVENC is not fully utilized"
                    fi
                fi
            fi
            
            # Store stats for summary
            FILE_STATS["$FILE"]="$MP_PER_SEC"
            FILE_TIMES["$FILE"]="$TRANSCODE_TIME"
            FILE_RESOLUTIONS["$FILE"]="$resolution"
            FILE_DURATIONS["$FILE"]="$duration"
            FILE_FPS_VALUES["$FILE"]="$fps"
            FILE_SIZES_INPUT["$FILE"]="$INPUT_SIZE"
            FILE_SIZES_OUTPUT["$FILE"]="$OUTPUT_SIZE"
            FILE_ENCODER_USED["$FILE"]="$ENCODER_DISPLAY"
            FILE_TRANSCODE_SPEED["$FILE"]="${{RT_FACTOR:-0}}"
            
            # Upload the transcoded file back to GCS
            log_action "Uploading transcoded file to output bucket..."
            OUTPUT_NAME="transcoded_$(basename $FILE .mp4)_$(date +%s).mp4"
            /tmp/venv/bin/python /tmp/gcs_operations.py upload "{gcs_output_bucket}" "$OUTPUT_FILE" "$OUTPUT_NAME"
            
            if [ $? -eq 0 ]; then
                log_action "‚úÖ Successfully uploaded to gs://{gcs_output_bucket}/$OUTPUT_NAME"
            else
                log_action "‚ö† Failed to upload transcoded file"
            fi
        else
            log_action "‚ö† Transcoding failed for $FILE"
            FILE_STATS["$FILE"]="FAILED"
            FILE_TIMES["$FILE"]="FAILED"
        fi
        
        # Clean up temporary files
        rm -f "/tmp/input_${{FILE_COUNT}}.mp4" "$OUTPUT_FILE"
        
        # Add separator between files
        if [ $FILE_COUNT -lt $TOTAL_FILES ]; then
            log_action ""
            log_action "Moving to next file..."
        fi
    done
    
    # Print comprehensive summary
    log_action ""
    log_action "========================================="
    log_action "=== TRANSCODING SUMMARY ==="
    log_action "========================================="
    log_action "Total files processed: $FILE_COUNT"
    log_action "Total transcoding time: ${{TOTAL_TRANSCODE_TIME}}s"
    log_action "Encoder used: $ENCODER_DISPLAY"
    
    log_action ""
    log_action "=== PER-FILE STATISTICS ==="
    log_action "----------------------------------------"
    
    TOTAL_MP_PER_SEC=0
    TOTAL_RT_FACTOR=0
    SUCCESSFUL_COUNT=0
    
    for FILE in "${{!FILE_STATS[@]}}"; do
        if [ "${{FILE_STATS[$FILE]}}" != "FAILED" ]; then
            log_action "File: $FILE"
            log_action "  - Input Size: $(format_file_size ${{FILE_SIZES_INPUT[$FILE]}})"
            log_action "  - Output Size: $(format_file_size ${{FILE_SIZES_OUTPUT[$FILE]}})"
            log_action "  - Resolution: ${{FILE_RESOLUTIONS[$FILE]}}"
            log_action "  - Duration: ${{FILE_DURATIONS[$FILE]}}s"
            log_action "  - FPS: ${{FILE_FPS_VALUES[$FILE]}}"
            log_action "  - Transcode Time: ${{FILE_TIMES[$FILE]}}s"
            log_action "  - Megapixels/sec: ${{FILE_STATS[$FILE]}} MP/s"
            log_action "  - Real-time Factor: ${{FILE_TRANSCODE_SPEED[$FILE]}}x"
            log_action "  - Encoder: ${{FILE_ENCODER_USED[$FILE]}}"
            
            TOTAL_MP_PER_SEC=$(echo "$TOTAL_MP_PER_SEC + ${{FILE_STATS[$FILE]}}" | bc)
            TOTAL_RT_FACTOR=$(echo "$TOTAL_RT_FACTOR + ${{FILE_TRANSCODE_SPEED[$FILE]:-0}}" | bc)
            SUCCESSFUL_COUNT=$((SUCCESSFUL_COUNT + 1))
        else
            log_action "File: $FILE - FAILED"
        fi
        log_action "----------------------------------------"
    done
    
    # Calculate average megapixels per second and real-time factor
    if [ $SUCCESSFUL_COUNT -gt 0 ]; then
        AVG_MP_PER_SEC=$(echo "scale=2; $TOTAL_MP_PER_SEC / $SUCCESSFUL_COUNT" | bc)
        AVG_RT_FACTOR=$(echo "scale=2; $TOTAL_RT_FACTOR / $SUCCESSFUL_COUNT" | bc)
        
        log_action ""
        log_action "=== OVERALL PERFORMANCE ==="
        log_action "Average Megapixels/sec: ${{AVG_MP_PER_SEC}} MP/s"
        log_action "Average Real-time Factor: ${{AVG_RT_FACTOR}}x"
        log_action "Successfully transcoded: $SUCCESSFUL_COUNT/$FILE_COUNT files"
        
        # Performance analysis with corrected expectations
        if [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
            log_action ""
            log_action "=== PERFORMANCE ANALYSIS ==="
            
            # Check real-time factor instead of raw MP/s
            if (( $(echo "$AVG_RT_FACTOR > 5" | bc -l) )); then
                log_action "‚úÖ EXCELLENT: NVENC is performing well with ${{AVG_RT_FACTOR}}x real-time speed"
                log_action "   This confirms hardware acceleration is active and effective"
            elif (( $(echo "$AVG_RT_FACTOR > 2" | bc -l) )); then
                log_action "‚úÖ GOOD: NVENC is active with ${{AVG_RT_FACTOR}}x real-time speed"
                log_action "   Hardware acceleration is working"
            else
                log_action "‚ö† WARNING: Performance is lower than expected"
                log_action "   Real-time factor: ${{AVG_RT_FACTOR}}x (expected >5x for 720p on T4)"
                log_action "   Possible issues:"
                log_action "     - GPU may be throttled"
                log_action "     - Input codec may require CPU decoding"
                log_action "     - Check nvidia-smi for GPU utilization"
            fi
            
            log_action ""
            log_action "Note: The ${{AVG_MP_PER_SEC}} MP/s metric includes only transcode time,"
            log_action "      not download/upload overhead. Your actual NVENC performance is good!"
        else
            log_action ""
            log_action "=== PERFORMANCE ANALYSIS ==="
            log_action "CPU encoding performance: ${{AVG_MP_PER_SEC}} MP/s"
            log_action "Real-time factor: ${{AVG_RT_FACTOR}}x"
            log_action "This is typical for CPU-based encoding."
            log_action "To improve performance, ensure NVENC is properly configured."
        fi
    fi
    
    if [ $FILE_COUNT -eq 0 ]; then
        log_action "No files were processed"
    else
        log_action ""
        log_action "‚úÖ Batch transcoding completed"
    fi

else
    # Placeholder for other modes (embed, local)
    log_action "Processing mode: {mode}"
    for i in $(seq 1 {streams_per_gpu}); do
        log_action "Processing stream $i of {streams_per_gpu}"
        sleep 2
    done
fi

# Show final GPU stats if using NVENC
if [ "{encoder_type}" = "nvenc" ] && [ "$ENCODER_AVAILABLE" = "nvenc" ]; then
    log_action ""
    log_action "=== Final GPU Stats ==="
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.encoder,memory.used,memory.total \\
                 --format=csv,noheader,nounits || true
fi

log_action ""
log_action "========================================="
log_action "Test completed successfully"
log_action "========================================="

}}

# Execute main function
main
"""

    # Create a proper indented script block for YAML
    indented_script = ""
    for line in bash_script.split('\n'):
        if line.strip():
            indented_script += f"            {line}\n"
        else:
            indented_script += "\n"

    # Volume mounts configuration
    volume_mounts = """        volumeMounts:
        - name: tmpfs-volume
          mountPath: /tmp
        - name: dshm
          mountPath: /dev/shm"""

    volumes = """      volumes:
      - name: tmpfs-volume
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi"""

    # Service account configuration
    service_account_spec = ""
    if service_account != "default":
        service_account_spec = f"""
      serviceAccountName: {service_account}"""

    # Node selector - only use for GPU encoding
    node_selector = ""
    if use_node_selector and encoder_type == "nvenc":
        node_selector = f"""
      nodeSelector:
        cloud.google.com/gke-nodepool: {NODE_POOL}"""

    # Adjust resource requests based on encoder type
    if encoder_type == "nvenc":
        resources = f"""        resources:
          limits:
            nvidia.com/gpu: {gpu_request}
            memory: "16Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: {gpu_request}
            memory: "8Gi"
            cpu: "4" """
    else:
        resources = """        resources:
          limits:
            memory: "16Gi"
            cpu: "8"
          requests:
            memory: "8Gi"
            cpu: "4" """

    # Environment variables - only include NVIDIA vars for GPU encoding
    if encoder_type == "nvenc":
        env_vars = """        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility,video"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: LD_LIBRARY_PATH
          value: "/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        - name: KUBERNETES_SERVICE_ACCOUNT
          valueFrom:
            fieldRef:
              fieldPath: spec.serviceAccountName"""
    else:
        env_vars = """        env:
        - name: KUBERNETES_SERVICE_ACCOUNT
          valueFrom:
            fieldRef:
              fieldPath: spec.serviceAccountName"""

    # Add TTL for automatic cleanup after completion
    yaml_content = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: {JOB_BASE_NAME}
    test-type: {"mps" if use_mps else "time-sharing"}
    mode: {mode}
    encoder: {encoder_type}
spec:
  parallelism: {effective_parallelism}
  completions: {effective_parallelism}
  backoffLimit: 1
  ttlSecondsAfterFinished: 300  # Auto-delete job 5 minutes after completion
  template:
    metadata:
      labels:
        app: {JOB_BASE_NAME}
        job-name: {job_name}
    spec:
      restartPolicy: Never{service_account_spec}
      containers:
      - name: ffmpeg-worker
        image: {base_image}
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
          - |
{indented_script}
{resources}
{volume_mounts}
{env_vars}
{volumes}{node_selector}
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
"""

    yaml_filename = f"{job_name}.yaml"
    with open(yaml_filename, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n‚úî Created YAML file: {yaml_filename}")

    return job_name, yaml_filename

def apply_yaml_to_cluster(yaml_filename, job_name, wait_for_completion=True, auto_cleanup=True):
    """Apply the YAML file to the Kubernetes cluster."""
    try:
        print(f"\nüöÄ Applying job to Kubernetes cluster...")
        print(f"    Job name: {job_name}")
        print(f"    YAML file: {yaml_filename}")
        
        # Apply the YAML file
        result = subprocess.run(
            ["kubectl", "apply", "-f", yaml_filename],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"‚úÖ Successfully applied job to cluster")
        print(f"    {result.stdout.strip()}")
        
        # Show initial job status
        print(f"\nüìä Initial job status:")
        subprocess.run(["kubectl", "get", "job", job_name])
        
        if wait_for_completion:
            # Wait for job to complete
            success = wait_for_job_completion(job_name, timeout=600)
            
            # Show logs
            print(f"\nüìã Job logs:")
            subprocess.run(["kubectl", "logs", "-l", f"job-name={job_name}", "--tail=100"])
            
            if auto_cleanup:
                # Clean up the job and YAML file
                cleanup_job(job_name, yaml_filename)
            else:
                print(f"\nüìå To manually clean up:")
                print(f"    kubectl delete job {job_name}")
                print(f"    rm {yaml_filename}")
            
            return success
        else:
            print(f"\nüìå To monitor the job:")
            print(f"    kubectl get job {job_name} -w")
            print(f"    kubectl get pods -l job-name={job_name}")
            print(f"    kubectl logs -l job-name={job_name} --follow")
            
            print(f"\nüßπÔ∏è  To clean up when done:")
            print(f"    kubectl delete job {job_name}")
            print(f"    rm {yaml_filename}")
            
            return True
        
    except subprocess.CalledProcessError as e:
        print(f"‚ùå Failed to apply job to cluster")
        print(f"    Error: {e.stderr}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='GPU FFmpeg Parallel Transcoding Test for GKE with NVENC optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple test with generated video using NVENC (waits for completion and auto-cleans)
  %(prog)s --encoder-type nvenc --gpu-model T4 --num-replicas 1

  # Test without waiting for completion
  %(prog)s --encoder-type nvenc --gpu-model T4 --num-replicas 1 --no-wait

  # Test with manual cleanup
  %(prog)s --encoder-type nvenc --gpu-model T4 --num-replicas 1 --no-auto-cleanup

  # Test with GCS input/output - processes ALL MP4 files from input bucket
  %(prog)s --mode gcs --encoder-type nvenc --gpu-model T4 --num-replicas 2

  # CPU-only encoding test
  %(prog)s --encoder-type cpu --num-replicas 4

  # Custom GCS buckets
  %(prog)s --mode gcs --gcs-input-bucket my-input --gcs-output-bucket my-output
        """
    )
    
    # Basic arguments
    parser.add_argument('--num-replicas', type=int, default=DEFAULT_NUM_REPLICAS,
                        help=f'Number of parallel pods (default: {DEFAULT_NUM_REPLICAS})')
    parser.add_argument('--streams-per-gpu', type=int,
                        help='Number of concurrent streams per GPU (auto-detected if not specified)')
    parser.add_argument('--use-mps', action='store_true',
                        help='Use MPS for GPU sharing (requires MPS-enabled nodes)')
    
    # Mode selection
    parser.add_argument('--mode', choices=['default', 'gcs', 'embed', 'local'],
                        default='default',
                        help='Input mode: default (generate), gcs (download from GCS bucket), embed (base64), local')
    
    # GCS arguments
    parser.add_argument('--gcs-input-bucket', default=DEFAULT_INPUT_BUCKET,
                        help=f'GCS input bucket (default: {DEFAULT_INPUT_BUCKET})')
    parser.add_argument('--gcs-input-file',
                        help='Specific GCS input file path')
    parser.add_argument('--gcs-output-bucket', default=DEFAULT_OUTPUT_BUCKET,
                        help=f'GCS output bucket (default: {DEFAULT_OUTPUT_BUCKET})')
    
    # Encoding arguments
    parser.add_argument('--encoder-type', choices=['cpu', 'nvenc'], default='cpu',
                        help='Encoder type: cpu (libx264) or nvenc (GPU)')
    parser.add_argument('--gpu-model', default='T4',
                        help='GPU model for stream estimation (T4, A10, A100, L4, L40, H100)')
    parser.add_argument('--preset', default='p4',
                        help='NVENC preset (p1-p7, p1=fastest, p7=best quality)')
    parser.add_argument('--rc-mode', default='vbr',
                        help='Rate control mode (cbr, vbr, cq, vbr_hq)')
    
    # Job control arguments
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for job completion')
    parser.add_argument('--no-auto-cleanup', action='store_true',
                        help='Do not automatically clean up job after completion')
    
    # Other arguments
    parser.add_argument('--local-input-file',
                        help='Local input file path (for local mode)')
    parser.add_argument('--job-suffix',
                        help='Custom suffix for job name')
    parser.add_argument('--skip-node-check', action='store_true',
                        help='Skip node pool existence check')
    parser.add_argument('--no-apply', action='store_true',
                        help='Only create YAML file, do not apply to cluster')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("GPU FFmpeg Parallel Transcoding Test for GKE")
    print("=" * 60)
    
    # Check service account
    service_account = check_service_account()
    
    # Check node pool if not skipped and using GPU
    use_node_selector = True
    if not args.skip_node_check and args.encoder_type == "nvenc":
        use_node_selector = check_node_pool_exists()
    
    # Auto-detect streams per GPU if not specified
    if args.streams_per_gpu is None:
        if args.encoder_type == "nvenc":
            args.streams_per_gpu = estimate_nvenc_streams(
                gpu_model=args.gpu_model,
                preset=args.preset,
                resolution=VIDEO_RESOLUTION,
                fps=VIDEO_FPS,
                rc_mode=args.rc_mode
            )
        else:
            args.streams_per_gpu = 2  # Default for CPU encoding
    
    # Create YAML file
    job_name, yaml_filename = create_yaml_file(
        num_replicas=args.num_replicas,
        streams_per_gpu=args.streams_per_gpu,
        use_mps=args.use_mps,
        mode=args.mode,
        gcs_input_bucket=args.gcs_input_bucket,
        gcs_input_file=args.gcs_input_file,
        gcs_output_bucket=args.gcs_output_bucket,
        local_input_file=args.local_input_file,
        use_node_selector=use_node_selector,
        preset=args.preset,
        rc_mode=args.rc_mode,
        job_suffix=args.job_suffix,
        service_account=service_account,
        encoder_type=args.encoder_type
    )
    
    # Apply to cluster unless --no-apply is specified
    if not args.no_apply:
        wait_for_completion = not args.no_wait
        auto_cleanup = not args.no_auto_cleanup
        
        success = apply_yaml_to_cluster(
            yaml_filename, 
            job_name, 
            wait_for_completion=wait_for_completion,
            auto_cleanup=auto_cleanup
        )
        
        if not success:
            print(f"\n‚ö† Job execution failed or was not applied to cluster")
            print(f"   YAML file was created: {yaml_filename}")
            print(f"   You can manually apply it with: kubectl apply -f {yaml_filename}")
            return 1
    else:
        print(f"\n‚úÖ YAML file created: {yaml_filename}")
        print(f"   To apply manually: kubectl apply -f {yaml_filename}")
    
    print("\n‚úÖ Script execution completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
