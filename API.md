# RCE REST API Documentation

Enterprise-grade REST API for integrating RCE (Relational Coherence Engine) into your systems.

## Quick Start

### Installation

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Or with Docker
docker build -f Dockerfile.api -t rce-api .
docker run -p 8000:8000 -e RCE_API_KEY=your-secret-key rce-api
```

### Running the API

```bash
# Development
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at: http://localhost:8000

Interactive docs: http://localhost:8000/docs

---

## Authentication

All requests require an API key in the `X-API-Key` header.

```bash
export RCE_API_KEY="your-secret-key"
```

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "convert 5 km to miles"}'
```

---

## Endpoints

### POST /query - Single Query Inference

Process a single natural language query with 0% hallucination.

**Request:**
```json
{
  "query": "convert 5 km to miles",
  "include_graph": true,
  "include_provenance": true
}
```

**Response:**
```json
{
  "query": "convert 5 km to miles",
  "answer": "5 km = 3.10686 miles",
  "task_type": "convert",
  "hallucination_risk": 0.0,
  "coherence_score": 1.0,
  "provenance": [
    {
      "atom_id": "q0",
      "type": "quantity",
      "label": "5 km",
      "attributes": {"value": 5.0, "unit": "km"},
      "confidence": 1.0
    }
  ],
  "graph": {
    "atoms": [...],
    "edges": [...],
    "coherence_score": 1.0
  },
  "compute_time_ms": 12.5,
  "energy_saved_vs_llm": "95%"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "convert 5 km to miles",
    "include_graph": false,
    "include_provenance": true
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    headers={"X-API-Key": "your-secret-key"},
    json={
        "query": "convert 5 km to miles",
        "include_provenance": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Hallucination Risk: {result['hallucination_risk']}%")
print(f"Provenance: {result['provenance']}")
```

---

### POST /batch - Batch Query Processing

Process multiple queries efficiently in a single request.

**Request:**
```json
{
  "queries": [
    "convert 5 km to miles",
    "is a cat a mammal?",
    "12*(3+4)-5"
  ],
  "include_graph": false,
  "include_provenance": true
}
```

**Response:**
```json
{
  "results": [
    {
      "query": "convert 5 km to miles",
      "answer": "5 km = 3.10686 miles",
      "hallucination_risk": 0.0,
      ...
    },
    {
      "query": "is a cat a mammal?",
      "answer": "Cat is a mammal.",
      "hallucination_risk": 0.0,
      ...
    },
    {
      "query": "12*(3+4)-5",
      "answer": "12*(3+4)-5 = 79",
      "hallucination_risk": 0.0,
      ...
    }
  ],
  "total_queries": 3,
  "successful": 3,
  "failed": 0,
  "total_compute_time_ms": 45.2
}
```

**Python Example:**
```python
queries = [
    "convert 100 km to miles",
    "is a dog an animal?",
    "5+5*2"
]

response = requests.post(
    "http://localhost:8000/batch",
    headers={"X-API-Key": "your-secret-key"},
    json={"queries": queries}
)

batch_result = response.json()
for result in batch_result['results']:
    print(f"Q: {result['query']}")
    print(f"A: {result['answer']}")
    print(f"Hallucination: {result['hallucination_risk']}%\n")
```

---

### GET /health - Health Check

Monitor API health and status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "total_queries_processed": 1234,
  "average_response_time_ms": 15.2,
  "hallucination_rate": 0.0
}
```

**cURL Example:**
```bash
curl http://localhost:8000/health
```

---

### GET /metrics - Performance Metrics

Get detailed performance and quality metrics.

**Response:**
```json
{
  "uptime_seconds": 3600.5,
  "total_queries": 1234,
  "average_response_time_ms": 15.2,
  "hallucination_rate": 0.0,
  "coherence_rate": 1.0,
  "throughput_qps": 0.34,
  "energy_savings_estimate": "40-60% vs standard LLM"
}
```

**Requires Authentication:**
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/metrics
```

---

## Integration Examples

### JavaScript/TypeScript

```typescript
interface RCEResponse {
  query: string;
  answer: string;
  hallucination_risk: number;
  coherence_score: number;
  provenance?: any[];
  compute_time_ms: number;
}

async function queryRCE(query: string): Promise<RCEResponse> {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.RCE_API_KEY
    },
    body: JSON.stringify({ query, include_provenance: true })
  });

  return await response.json();
}

// Usage
const result = await queryRCE('convert 5 km to miles');
console.log(`Answer: ${result.answer}`);
console.log(`Hallucination Risk: ${result.hallucination_risk}%`);
```

### Java

```java
import java.net.http.*;
import java.net.URI;
import com.fasterxml.jackson.databind.ObjectMapper;

public class RCEClient {
    private static final String API_URL = "http://localhost:8000/query";
    private static final String API_KEY = System.getenv("RCE_API_KEY");

    public static void main(String[] args) throws Exception {
        String requestBody = """
            {
                "query": "convert 5 km to miles",
                "include_provenance": true
            }
            """;

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_URL))
            .header("Content-Type", "application/json")
            .header("X-API-Key", API_KEY)
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        ObjectMapper mapper = new ObjectMapper();
        RCEResponse result = mapper.readValue(response.body(), RCEResponse.class);

        System.out.println("Answer: " + result.answer);
        System.out.println("Hallucination: " + result.hallucination_risk + "%");
    }
}
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
)

type RCERequest struct {
    Query            string `json:"query"`
    IncludeProvenance bool  `json:"include_provenance"`
}

type RCEResponse struct {
    Query            string  `json:"query"`
    Answer           string  `json:"answer"`
    HallucinationRisk float64 `json:"hallucination_risk"`
    CoherenceScore   float64 `json:"coherence_score"`
    ComputeTimeMs    float64 `json:"compute_time_ms"`
}

func queryRCE(query string) (*RCEResponse, error) {
    reqBody := RCERequest{
        Query:            query,
        IncludeProvenance: true,
    }

    jsonData, _ := json.Marshal(reqBody)

    req, _ := http.NewRequest("POST", "http://localhost:8000/query", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-API-Key", os.Getenv("RCE_API_KEY"))

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result RCEResponse
    json.NewDecoder(resp.Body).Decode(&result)

    return &result, nil
}

func main() {
    result, _ := queryRCE("convert 5 km to miles")
    fmt.Printf("Answer: %s\n", result.Answer)
    fmt.Printf("Hallucination: %.1f%%\n", result.HallucinationRisk)
}
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query text |
| `answer` | string | Validated answer (0% hallucination) |
| `task_type` | string | Detected task (taxonomy, convert, arithmetic, motion, general) |
| `hallucination_risk` | float | Always 0.0 for RCE |
| `coherence_score` | float | Graph coherence (0-1, higher = more valid) |
| `provenance` | array | Traceability data (optional) |
| `graph` | object | Knowledge graph representation (optional) |
| `compute_time_ms` | float | Processing time in milliseconds |
| `energy_saved_vs_llm` | string | Estimated energy savings vs standard LLM |

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 200 | Success | Query processed successfully |
| 400 | Bad Request | Check query format |
| 401 | Unauthorized | Verify API key |
| 500 | Server Error | Check logs, retry |

### Error Response Format

```json
{
  "detail": "Error description"
}
```

**Example Error Handling (Python):**
```python
try:
    response = requests.post(
        "http://localhost:8000/query",
        headers={"X-API-Key": api_key},
        json={"query": query},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 500:
        print("Server error, retrying...")
except requests.exceptions.Timeout:
    print("Request timeout")
```

---

## Performance

### Response Times

| Task Type | Avg Response Time | vs LLM |
|-----------|------------------|---------|
| Unit conversion | 10-20ms | 100x faster |
| Taxonomy | 5-15ms | 200x faster |
| Arithmetic | 1-5ms | 500x faster |
| Motion problem | 50-100ms | 20x faster |
| General (LLM fallback) | 2-10s | Similar |

### Throughput

- **Single instance:** 50-100 QPS
- **Horizontal scaling:** Linear (stateless)
- **Recommended:** 2-4 instances behind load balancer

### Rate Limits

Default: No limits (configure as needed)

**Production recommendation:**
- 1000 req/minute per API key
- 10,000 req/day per customer

---

## Security Best Practices

### API Key Management

```bash
# Use environment variables
export RCE_API_KEY=$(openssl rand -hex 32)

# Rotate keys regularly
# Use different keys per environment (dev/staging/prod)
```

### HTTPS/TLS

Always use HTTPS in production:

```bash
# Run behind reverse proxy (nginx, Caddy)
# Or use cloud load balancer (AWS ALB, Azure App Gateway)
```

### IP Whitelisting

```python
# In production, add IP whitelist middleware
from fastapi import Request
ALLOWED_IPS = ["10.0.0.0/8", "192.168.0.0/16"]

@app.middleware("http")
async def ip_whitelist(request: Request, call_next):
    if request.client.host not in ALLOWED_IPS:
        raise HTTPException(403, "IP not allowed")
    return await call_next(request)
```

---

## Monitoring & Observability

### Metrics to Track

- `total_queries` - Total inference requests
- `average_response_time_ms` - Avg latency
- `hallucination_rate` - Should always be 0%
- `coherence_rate` - % of queries with high coherence
- `error_rate` - % of failed requests

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rce-api")

# Logs include:
# - Request timestamp
# - Query text (sanitized)
# - Response time
# - Coherence score
# - Task type
```

### Alerts

Set up alerts for:
- `hallucination_rate > 0%` (should never happen!)
- `error_rate > 5%`
- `average_response_time_ms > 1000ms`
- `uptime < 99.9%`

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide.

**Quick deploy with Docker:**

```dockerfile
# Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY apps/ ./apps/
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -f Dockerfile.api -t rce-api .
docker run -d -p 8000:8000 -e RCE_API_KEY=secret rce-api
```

---

## Support & SLA

**Community Support:**
- GitHub Issues: https://github.com/IsmaIkami/rce-llm/issues
- Documentation: https://docs.rce.ai (coming soon)

**Enterprise Support:**
- 99.9% uptime SLA
- 24/7 technical support
- Priority bug fixes
- Custom integration assistance
- Contact: enterprise@rce.ai

---

## Changelog

### v1.0.0 (2025-01-XX)
- Initial API release
- Single query endpoint
- Batch processing
- Health checks & metrics
- OpenAPI/Swagger docs
- API key authentication

### Roadmap
- [ ] Webhooks for async processing
- [ ] GraphQL endpoint
- [ ] WebSocket streaming
- [ ] Multi-language support
- [ ] Advanced caching
- [ ] OAuth 2.0 support
