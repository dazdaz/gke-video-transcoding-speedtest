#!/bin/bash

#. **Grant Artifact Registry Reader permissions** to:
#   - The node pool service account
#   - The Workload Identity service account (if using)

# Set your variables
export PROJECT_ID="my-playground"
export REGION="us-central1"
export CLUSTER_NAME="tp"  # Replace with your cluster name
export ZONE="us-central1-a"  # Replace with your cluster zone

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Get the node pool service account
NODE_SA=$(gcloud container node-pools describe transcoder-gpunode-pool-mps \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --format="value(config.serviceAccount)" 2>/dev/null || echo "default")

# If default, use the Compute Engine default service account
if [ "$NODE_SA" == "default" ] || [ -z "$NODE_SA" ]; then
    NODE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    echo "Using default Compute Engine service account: $NODE_SA"
else
    echo "Using custom service account: $NODE_SA"
fi

# Grant Artifact Registry Reader role
echo "Granting Artifact Registry Reader role to $NODE_SA..."
gcloud artifacts repositories add-iam-policy-binding ffmpeg-nvidia \
    --location=$REGION \
    --member="serviceAccount:$NODE_SA" \
    --role="roles/artifactregistry.reader" \
    --project=$PROJECT_ID

# Configure Docker/containerd authentication for Artifact Registry
echo "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# If using Workload Identity (from your gke-tp-service-account)
if kubectl get serviceaccount gke-tp-service-account -n default &>/dev/null; then
    echo "Found gke-tp-service-account, checking Workload Identity binding..."
    
    # Get the GCP service account email
    GCP_SA=$(kubectl get serviceaccount gke-tp-service-account -n default \
        -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}')
    
    if [ ! -z "$GCP_SA" ]; then
        echo "Granting Artifact Registry Reader to Workload Identity SA: $GCP_SA"
        gcloud artifacts repositories add-iam-policy-binding ffmpeg-nvidia \
            --location=$REGION \
            --member="serviceAccount:$GCP_SA" \
            --role="roles/artifactregistry.reader" \
            --project=$PROJECT_ID
    fi
fi

echo "‚úÖ Artifact Registry access configured!"
echo ""
echo "You can now use the image in your jobs:"
echo "  ${REGION}-docker.pkg.dev/${PROJECT_ID}/ffmpeg-nvidia/ffmpeg-nvidia-nvenc:latest"
