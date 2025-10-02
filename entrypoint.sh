#!/bin/bash

# entrypoint.sh
# Enhanced entrypoint script with better error handling and performance metrics

set -e

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if running in test mode
check_test_mode() {
    # If no arguments or first argument starts with -, we're in test mode
    if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
        return 0
    fi
    return 1
}

# Test mode - just verify FFmpeg installation
if check_test_mode "$@"; then
    log_message "Running in test/verification mode"
    log_message "Checking FFmpeg installation..."
    
    # Check FFmpeg version
    ffmpeg -version
    
    # Check for NVENC support
    log_message ""
    log_message "Checking for NVENC encoder support..."
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
        log_message "‚úÖ h264_nvenc encoder available"
    else
        log_message "‚ùå h264_nvenc encoder NOT available"
    fi
    
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc; then
        log_message "‚úÖ hevc_nvenc encoder available"
    else
        log_message "‚ùå hevc_nvenc encoder NOT available"
    fi
    
    # Check NVIDIA environment if available
    log_message ""
    log_message "Checking NVIDIA environment..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || log_message "nvidia-smi failed"
    else
        log_message "nvidia-smi not available (running on CPU-only node?)"
    fi
    
    # Check for Google Cloud SDK
    log_message ""
    log_message "Checking Google Cloud SDK..."
    if command -v gcloud &> /dev/null; then
        gcloud version
    else
        log_message "gcloud not available"
    fi
    
    if command -v gsutil &> /dev/null; then
        log_message "‚úÖ gsutil available"
    else
        log_message "‚ùå gsutil not available"
    fi
    
    log_message ""
    log_message "Test mode complete. Container is ready for transcoding."
    exit 0
fi

# Production mode - process video files
INPUT_FILE=$1
OUTPUT_FILE=$2
shift 2
FFMPEG_ARGS=("$@")

# Get bucket names from environment variables
SOURCE_BUCKET=${SOURCE_BUCKET:-transcode-preprocessing-bucket}
TARGET_BUCKET=${TARGET_BUCKET:-transcode-postprocessing-bucket}

# Define local processing directory
WORK_DIR="/tmp/transcode"
mkdir -p "${WORK_DIR}"
LOCAL_INPUT="${WORK_DIR}/${INPUT_FILE}"
LOCAL_OUTPUT="${WORK_DIR}/${OUTPUT_FILE}"

log_message "==============================================="
log_message "Starting transcode job"
log_message "Source: gs://${SOURCE_BUCKET}/${INPUT_FILE}"
log_message "Target: gs://${TARGET_BUCKET}/${OUTPUT_FILE}"
log_message "==============================================="

# Check NVIDIA GPU availability
log_message ""
log_message "Checking GPU availability..."
if nvidia-smi &>/dev/null; then
    log_message "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    USE_GPU=true
else
    log_message "No GPU detected, will use CPU encoding"
    USE_GPU=false
fi

# Step 1: Download the input file from GCS
log_message ""
log_message "Step 1: Downloading input file from GCS..."
log_message "Command: gsutil -m cp gs://${SOURCE_BUCKET}/${INPUT_FILE} ${LOCAL_INPUT}"
DOWNLOAD_START=$(date +%s.%N)

if ! gsutil -m cp "gs://${SOURCE_BUCKET}/${INPUT_FILE}" "${LOCAL_INPUT}"; then
    log_message "ERROR: Failed to download input file from GCS"
    exit 1
fi

DOWNLOAD_END=$(date +%s.%N)
DOWNLOAD_TIME=$(echo "$DOWNLOAD_END - $DOWNLOAD_START" | bc)
FILE_SIZE_MB=$(du -m "${LOCAL_INPUT}" | cut -f1)
log_message "Download complete in ${DOWNLOAD_TIME} seconds. File size: ${FILE_SIZE_MB} MB"

# Step 2: Extract video information
log_message ""
log_message "Step 2: Extracting video information..."
if [ ! -f "${LOCAL_INPUT}" ]; then
    log_message "ERROR: Input file not found at ${LOCAL_INPUT}"
    exit 1
fi

# Extract video information using ffprobe
VIDEO_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration,codec_name -of json "${LOCAL_INPUT}" 2>/dev/null || echo "{}")

if [ "$VIDEO_INFO" != "{}" ]; then
    VIDEO_WIDTH=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('width', 'unknown'))" 2>/dev/null || echo "unknown")
    VIDEO_HEIGHT=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('height', 'unknown'))" 2>/dev/null || echo "unknown")
    FRAME_RATE_STR=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('r_frame_rate', '0/1'))" 2>/dev/null || echo "0/1")
    TOTAL_FRAMES=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('nb_frames', 'unknown'))" 2>/dev/null || echo "unknown")
    DURATION=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('duration', 'unknown'))" 2>/dev/null || echo "unknown")
    CODEC_NAME=$(echo "$VIDEO_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('streams', [{}])[0].get('codec_name', 'unknown'))" 2>/dev/null || echo "unknown")
    
    # Calculate frame rate as decimal
    if [ "$FRAME_RATE_STR" != "0/1" ] && [ "$FRAME_RATE_STR" != "unknown" ]; then
        FRAME_RATE=$(echo "$FRAME_RATE_STR" | python3 -c "import sys; parts=sys.stdin.read().strip().split('/'); print(float(parts[0])/float(parts[1]) if len(parts)==2 and parts[1]!='0' else 0)" 2>/dev/null || echo "unknown")
    else
        FRAME_RATE="unknown"
    fi
else
    VIDEO_WIDTH="unknown"
    VIDEO_HEIGHT="unknown"
    FRAME_RATE="unknown"
    TOTAL_FRAMES="unknown"
    DURATION="unknown"
    CODEC_NAME="unknown"
fi

log_message "Input video information:"
log_message "  Codec: ${CODEC_NAME}"
log_message "  Resolution: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}"
log_message "  Frame rate: ${FRAME_RATE} fps"
log_message "  Total frames: ${TOTAL_FRAMES}"
log_message "  Duration: ${DURATION} seconds"

# Step 3: Determine decoder and encoder
log_message ""
log_message "Step 3: Configuring decoder and encoder..."

# Determine decoder based on input codec and GPU availability
DECODER_ARGS=""
if [ "$USE_GPU" = true ]; then
    case "$CODEC_NAME" in
        h264)
            if ffmpeg -decoders 2>/dev/null | grep -q h264_cuvid; then
                DECODER_ARGS="-c:v h264_cuvid"
                log_message "Using GPU decoder: h264_cuvid"
            fi
            ;;
        hevc|h265)
            if ffmpeg -decoders 2>/dev/null | grep -q hevc_cuvid; then
                DECODER_ARGS="-c:v hevc_cuvid"
                log_message "Using GPU decoder: hevc_cuvid"
            fi
            ;;
        *)
            log_message "Using CPU decoder for codec: ${CODEC_NAME}"
            ;;
    esac
else
    log_message "Using CPU decoder"
fi

# Build final FFmpeg command
if [ "$USE_GPU" = true ] && ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc; then
    # GPU encoding with NVENC
    log_message "Using GPU encoder: h264_nvenc"
    FFMPEG_CMD="ffmpeg -y ${DECODER_ARGS} -i '${LOCAL_INPUT}' ${FFMPEG_ARGS[@]} -c:v h264_nvenc -preset p7 '${LOCAL_OUTPUT}'"
else
    # CPU encoding fallback
    log_message "Using CPU encoder: libx264"
    FFMPEG_CMD="ffmpeg -y -i '${LOCAL_INPUT}' ${FFMPEG_ARGS[@]} -c:v libx264 -preset slow '${LOCAL_OUTPUT}'"
fi

# Step 4: Run FFmpeg transcoding
log_message ""
log_message "Step 4: Starting FFmpeg transcoding..."
log_message "Command: ${FFMPEG_CMD}"

TRANSCODE_START=$(date +%s.%N)
TRANSCODE_START_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
log_message "Transcoding started at: ${TRANSCODE_START_READABLE}"

# Execute ffmpeg
eval ${FFMPEG_CMD} 2>&1 | tee /tmp/ffmpeg.log
FFMPEG_EXIT_CODE=${PIPESTATUS[0]}

TRANSCODE_END=$(date +%s.%N)
TRANSCODE_END_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
TRANSCODE_DURATION=$(echo "$TRANSCODE_END - $TRANSCODE_START" | bc)

log_message "Transcoding ended at: ${TRANSCODE_END_READABLE}"
log_message "TRANSCODING DURATION: ${TRANSCODE_DURATION} seconds"

if [ $FFMPEG_EXIT_CODE -ne 0 ]; then
    log_message "ERROR: FFmpeg transcoding failed with exit code ${FFMPEG_EXIT_CODE}"
    rm -f "${LOCAL_INPUT}"
    exit $FFMPEG_EXIT_CODE
fi

OUTPUT_SIZE_MB=$(du -m "${LOCAL_OUTPUT}" | cut -f1)
log_message "Transcoding complete. Output file size: ${OUTPUT_SIZE_MB} MB"

# Calculate performance metrics
if [ "$VIDEO_WIDTH" != "unknown" ] && [ "$VIDEO_HEIGHT" != "unknown" ] && \
   [ "$FRAME_RATE" != "unknown" ] && [ "$DURATION" != "unknown" ] && \
   [ "$DURATION" != "0" ]; then
    
    TOTAL_PIXELS=$(echo "$VIDEO_WIDTH * $VIDEO_HEIGHT" | bc)
    MEGAPIXELS=$(echo "scale=2; $TOTAL_PIXELS / 1000000" | bc)
    
    if [ "$TOTAL_FRAMES" == "unknown" ] && [ "$FRAME_RATE" != "0" ]; then
        TOTAL_FRAMES=$(echo "$DURATION * $FRAME_RATE" | bc | cut -d. -f1)
    fi
    
    if [ "$TOTAL_FRAMES" != "unknown" ] && [ "$TOTAL_FRAMES" != "0" ]; then
        TOTAL_MEGAPIXELS=$(echo "scale=2; $MEGAPIXELS * $TOTAL_FRAMES" | bc)
        MEGAPIXELS_PER_SECOND=$(echo "scale=2; $TOTAL_MEGAPIXELS / $TRANSCODE_DURATION" | bc)
        FPS_ENCODING=$(echo "scale=2; $TOTAL_FRAMES / $TRANSCODE_DURATION" | bc)
        REALTIME_FACTOR=$(echo "scale=2; $DURATION / $TRANSCODE_DURATION" | bc)
        
        log_message ""
        log_message "=== PERFORMANCE METRICS ==="
        log_message "Video resolution: ${VIDEO_WIDTH}x${VIDEO_HEIGHT} (${MEGAPIXELS} MP per frame)"
        log_message "Total frames processed: ${TOTAL_FRAMES}"
        log_message "Total megapixels processed: ${TOTAL_MEGAPIXELS} MP"
        log_message "Transcoding time: ${TRANSCODE_DURATION} seconds"
        log_message "MEGAPIXELS PER SECOND: ${MEGAPIXELS_PER_SECOND} MP/s"
        log_message "Encoding speed: ${FPS_ENCODING} fps"
        log_message "Real-time factor: ${REALTIME_FACTOR}x"
        log_message "=========================="
    fi
fi

# Step 5: Verify output file
log_message ""
log_message "Step 5: Verifying output file..."
if [ ! -f "${LOCAL_OUTPUT}" ]; then
    log_message "ERROR: Output file not found at ${LOCAL_OUTPUT}"
    rm -f "${LOCAL_INPUT}"
    exit 1
fi

ffprobe -v error -show_format -show_streams "${LOCAL_OUTPUT}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log_message "WARNING: Output file may be corrupted"
fi

# Step 6: Upload output file to GCS
log_message ""
log_message "Step 6: Uploading output file to GCS..."
log_message "Command: gsutil -m cp ${LOCAL_OUTPUT} gs://${TARGET_BUCKET}/${OUTPUT_FILE}"

UPLOAD_START=$(date +%s.%N)
if ! gsutil -m cp "${LOCAL_OUTPUT}" "gs://${TARGET_BUCKET}/${OUTPUT_FILE}"; then
    log_message "ERROR: Failed to upload output file to GCS"
    rm -f "${LOCAL_INPUT}" "${LOCAL_OUTPUT}"
    exit 1
fi
UPLOAD_END=$(date +%s.%N)
UPLOAD_TIME=$(echo "$UPLOAD_END - $UPLOAD_START" | bc)
log_message "Upload complete in ${UPLOAD_TIME} seconds"

# Step 7: Clean up
log_message ""
log_message "Step 7: Cleaning up local files..."
rm -f "${LOCAL_INPUT}" "${LOCAL_OUTPUT}"
log_message "Cleanup complete"

# Final summary
TOTAL_TIME=$(echo "$DOWNLOAD_TIME + $TRANSCODE_DURATION + $UPLOAD_TIME" | bc)

log_message ""
log_message "==============================================="
log_message "TRANSCODE JOB SUMMARY"
log_message "==============================================="
log_message "Input: gs://${SOURCE_BUCKET}/${INPUT_FILE}"
log_message "Output: gs://${TARGET_BUCKET}/${OUTPUT_FILE}"
log_message "Input size: ${FILE_SIZE_MB} MB"
log_message "Output size: ${OUTPUT_SIZE_MB} MB"
log_message "Compression ratio: $(echo "scale=2; $OUTPUT_SIZE_MB / $FILE_SIZE_MB" | bc)"
log_message "Download time: ${DOWNLOAD_TIME} seconds"
log_message "TRANSCODE TIME: ${TRANSCODE_DURATION} seconds"
log_message "Upload time: ${UPLOAD_TIME} seconds"
log_message "Total job time: ${TOTAL_TIME} seconds"
if [ "${MEGAPIXELS_PER_SECOND:-}" != "" ]; then
    log_message "Performance: ${MEGAPIXELS_PER_SECOND} megapixels/second"
fi
log_message "==============================================="
log_message "Transcode job completed successfully!"
log_message "==============================================="

exit 0
