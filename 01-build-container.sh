#!/bin/bash

# 01-build-container.sh
# Build and push FFmpeg container with NVIDIA support to Artifact Registry

set -e

# Configuration
export PROJECT_ID="${PROJECT_ID:-my-playground}"
export REGION="${REGION:-us-central1}"
export REPOSITORY_NAME="ffmpeg-nvidia"
export IMAGE_NAME="ffmpeg-nvidia-nvenc"
export IMAGE_TAG="latest"

# Full image path
export FULL_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "==================================================="
echo "Building FFmpeg Container with NVIDIA NVENC Support"
echo "==================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY_NAME}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Full Path: ${FULL_IMAGE_PATH}"
echo "==================================================="

# Set the gcloud project
echo ""
echo "Setting gcloud project..."
gcloud config set project ${PROJECT_ID}

# Enable necessary APIs
echo ""
echo "Enabling required services..."
gcloud services enable cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    container.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo ""
echo "Creating/verifying Artifact Registry repository..."
if gcloud artifacts repositories describe ${REPOSITORY_NAME} \
    --location=${REGION} >/dev/null 2>&1; then
    echo "Repository '${REPOSITORY_NAME}' already exists."
else
    echo "Creating repository '${REPOSITORY_NAME}'..."
    gcloud artifacts repositories create ${REPOSITORY_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="FFmpeg with NVIDIA GPU support for transcoding"
fi

# Configure Docker/Podman authentication for Artifact Registry
echo ""
echo "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Check if we should use Cloud Build or local build
if [ "${BUILD_LOCAL}" == "true" ]; then
    echo ""
    echo "Building locally with Docker/Podman..."
    
    # Check if podman is available, otherwise use docker
    if command -v podman &> /dev/null; then
        BUILD_CMD="podman"
        echo "Using Podman for local build"
    elif command -v docker &> /dev/null; then
        BUILD_CMD="docker"
        echo "Using Docker for local build"
    else
        echo "ERROR: Neither Docker nor Podman found. Please install one of them."
        exit 1
    fi
    
    # Build the image locally
    echo "Building image..."
    ${BUILD_CMD} build -t ${FULL_IMAGE_PATH} .
    
    # Push the image
    echo "Pushing image to Artifact Registry..."
    ${BUILD_CMD} push ${FULL_IMAGE_PATH}
else
    # Use Cloud Build (recommended for consistency)
    echo ""
    echo "Submitting build to Cloud Build..."
    
    # Create a cloudbuild.yaml if it doesn't exist
    if [ ! -f "cloudbuild.yaml" ]; then
        echo "Creating cloudbuild.yaml..."
        cat > cloudbuild.yaml <<EOF
steps:
  # Build the Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${FULL_IMAGE_PATH}', '.']
    
  # Push the Docker image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${FULL_IMAGE_PATH}']

# Use machine type with more CPU for faster builds
options:
  machineType: 'E2_HIGHCPU_32'
  logging: LEGACY

# Tag the image
images:
  - '${FULL_IMAGE_PATH}'

# Timeout for the build (30 minutes should be enough)
timeout: '1800s'
EOF
    fi
    
    # Submit the build
    gcloud builds submit --config=cloudbuild.yaml .
fi

# Verify the image was pushed successfully
echo ""
echo "Verifying image in Artifact Registry..."
if gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME} \
    --filter="package=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}" \
    --format="table(package,tags,createTime)" | grep -q ${IMAGE_TAG}; then
    echo "[SUCCESS] Image successfully pushed to Artifact Registry!"
    echo ""
    echo "Image URI: ${FULL_IMAGE_PATH}"
    echo ""
    echo "You can now use this image in your Kubernetes deployments."
else
    echo "[FAILED] Failed to verify image in Artifact Registry"
    exit 1
fi

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo ""
echo "To test the image locally:"
echo "  docker run --rm --gpus all ${FULL_IMAGE_PATH} ffmpeg -version"
echo ""
echo "To use in Kubernetes:"
echo "  Update your deployment YAML with image: ${FULL_IMAGE_PATH}"
echo ""
