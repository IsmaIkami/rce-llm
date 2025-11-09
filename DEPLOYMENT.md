# RCE Deployment Guide

This guide covers deployment options for the Relational Coherence Engine (RCE).

## Quick Start (Docker)

### Prerequisites
- Docker 20.10+
- Docker Compose 1.29+ (optional, recommended)

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/IsmaIkami/rce-llm.git
cd rce-llm

# Start RCE
docker-compose up -d

# View logs
docker-compose logs -f

# Stop RCE
docker-compose down
```

Access at: http://localhost:8501

### Option 2: Docker Run

```bash
# Build image
docker build -t rce-engine .

# Run container
docker run -d \
  --name rce-engine \
  -p 8501:8501 \
  --restart unless-stopped \
  rce-engine

# View logs
docker logs -f rce-engine

# Stop container
docker stop rce-engine
```

---

## Production Deployment

### AWS Deployment

#### Option A: ECS (Elastic Container Service)

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name rce-engine
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t rce-engine .
docker tag rce-engine:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rce-engine:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rce-engine:latest

# 2. Create ECS task definition (see aws/ecs-task-definition.json)

# 3. Create ECS service
aws ecs create-service \
  --cluster rce-cluster \
  --service-name rce-service \
  --task-definition rce-engine \
  --desired-count 2 \
  --launch-type FARGATE
```

#### Option B: EC2 Instance

```bash
# SSH to EC2 instance
ssh -i key.pem ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy RCE
git clone https://github.com/IsmaIkami/rce-llm.git
cd rce-llm
docker-compose up -d
```

Configure security group to allow port 8501.

---

### Azure Deployment

#### Azure Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name rce-rg --location eastus

# Create container instance
az container create \
  --resource-group rce-rg \
  --name rce-engine \
  --image <your-docker-hub>/rce-engine:latest \
  --dns-name-label rce-engine-demo \
  --ports 8501

# Get public IP
az container show \
  --resource-group rce-rg \
  --name rce-engine \
  --query ipAddress.fqdn
```

---

### Google Cloud Platform

#### Cloud Run

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/rce-engine
gcloud run deploy rce-engine \
  --image gcr.io/<project-id>/rce-engine \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

---

### Kubernetes Deployment

```bash
# Create deployment
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Get external IP
kubectl get services rce-service
```

**k8s/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rce-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rce-engine
  template:
    metadata:
      labels:
        app: rce-engine
    spec:
      containers:
      - name: rce-engine
        image: <your-registry>/rce-engine:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
```

**k8s/service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rce-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: rce-engine
```

---

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `STREAMLIT_SERVER_PORT` | Port for Streamlit | 8501 | No |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | 0.0.0.0 | No |
| `HF_TOKEN` | HuggingFace API token (for LLM fallback) | None | No |
| `HF_MODEL` | HuggingFace model name | mistralai/Mistral-7B-Instruct-v0.2 | No |

---

## Scaling Considerations

### Horizontal Scaling

RCE is stateless and can be horizontally scaled:

```bash
# Docker Compose
docker-compose up -d --scale rce-engine=3

# Kubernetes
kubectl scale deployment rce-engine --replicas=5
```

### Resource Requirements

**Minimum:**
- CPU: 1 core
- Memory: 512MB
- Disk: 1GB

**Recommended (Production):**
- CPU: 2-4 cores
- Memory: 2-4GB
- Disk: 5GB

**With LLM Fallback:**
- CPU: 4-8 cores
- Memory: 8-16GB
- GPU: Optional (NVIDIA T4 or better)

---

## Monitoring & Logging

### Health Check

```bash
curl http://localhost:8501/_stcore/health
```

### Metrics

RCE exposes metrics for monitoring:
- Request count
- Response time
- Hallucination rate (always 0%)
- Coherence scores
- Compute time savings

### Logging

Logs are written to stdout/stderr and can be collected:

```bash
# Docker
docker logs -f rce-engine

# Kubernetes
kubectl logs -f deployment/rce-engine
```

---

## Security

### HTTPS/TLS

For production, use a reverse proxy (nginx, Caddy) or cloud load balancer:

**nginx example:**
```nginx
server {
    listen 443 ssl;
    server_name rce.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Authentication

For enterprise deployments, add authentication layer:
- OAuth 2.0 / OIDC
- API key validation
- IP whitelisting

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs rce-engine

# Common issues:
# - Port 8501 already in use: Change port mapping
# - Missing dependencies: Rebuild image
```

### High memory usage

```bash
# Set memory limits
docker run --memory=2g rce-engine

# Or in docker-compose.yml:
services:
  rce-engine:
    mem_limit: 2g
```

### Slow performance

- Enable caching for repeated queries
- Use deterministic resolvers (avoid LLM fallback)
- Scale horizontally
- Add CDN/cache layer

---

## Support

For deployment assistance:
- GitHub Issues: https://github.com/IsmaIkami/rce-llm/issues
- Email: [support email]
- Enterprise support: [contact for SLA]
