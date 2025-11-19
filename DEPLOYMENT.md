# Render.com Deployment Guide

## Quick Deploy (5 Minutes)

Your Network Intrusion Detection System is now ready to deploy to Render.com with **infrastructure-as-code**.

---

## Prerequisites

- GitHub account with repository: `https://github.com/byessilyurt/network-intrusion-detection`
- Render.com account (sign up free at https://render.com)

---

## Deployment Steps

### Step 1: Sign Up / Sign In to Render.com

1. Go to https://render.com
2. Click **"Get Started"** or **"Sign In"**
3. Choose **"Sign in with GitHub"** for easiest setup

### Step 2: Create New Blueprint

1. From Render Dashboard, click **"New +"** → **"Blueprint"**
2. Connect your GitHub account if not already connected
3. Search for: `network-intrusion-detection`
4. Select your repository
5. Click **"Connect"**

### Step 3: Configure Blueprint

Render will automatically detect the `render.yaml` file.

**Review the configuration:**
- **Service 1: nids-api** (FastAPI Backend)
  - Type: Web Service
  - Port: 8000
  - Plan: Free
  - Health Check: `/health`

- **Service 2: nids-dashboard** (Streamlit Frontend)
  - Type: Web Service
  - Port: 8501
  - Plan: Free
  - Health Check: `/_stcore/health`

### Step 4: Update API URL for Dashboard

**IMPORTANT**: After the API service deploys, you need to update the dashboard's API_URL:

1. Wait for `nids-api` to finish deploying (~3-5 minutes)
2. Copy the API URL (will be like: `https://nids-api.onrender.com`)
3. Go to `nids-dashboard` service
4. Click **"Environment"** in the left sidebar
5. Find the `API_URL` environment variable
6. Update it to your actual API URL: `https://nids-api.onrender.com`
7. Click **"Save Changes"**
8. The dashboard will automatically redeploy

### Step 5: Deploy

1. Click **"Apply"** to deploy both services
2. Wait for build and deployment (~5-7 minutes total)
3. Both services will show "Live" status when ready

---

## Accessing Your Deployed App

### API Service
- **URL**: `https://nids-api.onrender.com`
- **Swagger Docs**: `https://nids-api.onrender.com/docs`
- **Health Check**: `https://nids-api.onrender.com/health`

### Dashboard Service
- **URL**: `https://nids-dashboard.onrender.com`
- Interactive web interface for analyzing network flows

---

## Testing Your Deployment

### Test the API

```bash
# Health check
curl https://nids-api.onrender.com/health

# Get model info
curl https://nids-api.onrender.com/model/info

# Test prediction (example)
curl -X POST "https://nids-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      ...
    }
  }'
```

### Test the Dashboard

1. Open `https://nids-dashboard.onrender.com`
2. Navigate to "Sample Data" tab
3. Click "Load Sample" → "DoS Attack"
4. Download the CSV
5. Go to "Single Flow Analysis" tab
6. Upload the CSV
7. Click "Analyze Flow"
8. View detection results with SHAP explanation

---

## Important Notes

### Free Tier Limitations

- **Cold Starts**: Services sleep after 15 minutes of inactivity
- **First Request**: May take 30-60 seconds to wake up
- **Subsequent Requests**: Fast (~50ms response time)
- **Recommendation**: Upgrade to paid tier ($7/month per service) for instant responses

### Upgrade to Paid Tier (Optional)

1. Go to each service → "Settings"
2. Under "Instance Type", click "Change"
3. Select "Starter" ($7/month)
4. Click "Save"
5. Service will restart with always-on availability

### Resource Usage

- **API**: ~200MB RAM, minimal CPU
- **Dashboard**: ~150MB RAM, minimal CPU
- **Storage**: Models and results included in Docker image
- **Total Cost (Free Tier)**: $0/month
- **Total Cost (Starter)**: $14/month (both services)

---

## Monitoring Your Deployment

### View Logs

1. Select service (nids-api or nids-dashboard)
2. Click "Logs" tab
3. Real-time logs will appear

### Check Metrics

1. Select service
2. Click "Metrics" tab
3. View:
   - CPU usage
   - Memory usage
   - Request count
   - Response time

### Health Checks

- Both services have automatic health checks every 30 seconds
- If health check fails 3 times, service automatically restarts
- API: `GET /health`
- Dashboard: `GET /_stcore/health`

---

## Troubleshooting

### Issue: Dashboard shows "API Offline"

**Solution:**
1. Check if API service is running (should show "Live")
2. Verify `API_URL` environment variable in dashboard
3. Ensure API URL doesn't have trailing slash
4. Check API logs for errors

### Issue: "Service Unavailable" or slow first request

**Solution:**
- This is expected on Free tier (cold starts)
- First request wakes up the service (~30-60s)
- Subsequent requests are fast
- Upgrade to paid tier for instant responses

### Issue: Build fails

**Solution:**
1. Check build logs for specific error
2. Common issues:
   - Missing dependencies → Check requirements.txt
   - Large model files → Ensure models/ directory exists
   - Docker build timeout → Contact Render support

### Issue: Out of Memory

**Solution:**
- Free tier has 512MB RAM limit
- API loads full OCSVM model + SHAP (~150MB)
- If OOM occurs:
  1. Upgrade to Starter tier (512MB → 2GB RAM)
  2. Or reduce SHAP_NSAMPLES environment variable

---

## Custom Domain (Optional)

1. Purchase domain from registrar (Namecheap, GoDaddy, etc.)
2. In Render Dashboard, select service
3. Click "Settings" → "Custom Domains"
4. Click "Add Custom Domain"
5. Enter your domain (e.g., `nids.yourdomain.com`)
6. Add CNAME record to your DNS:
   - **Name**: `nids`
   - **Value**: `nids-api.onrender.com` (or dashboard URL)
7. Wait for DNS propagation (~1-24 hours)

---

## Environment Variables

### API Service (`nids-api`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API port (auto-set by Render) |
| `MODEL_PATH` | `/app/models/ocsvm_200k.pkl` | Path to OCSVM model |
| `DATA_PATH` | `/app/data/raw/Monday-WorkingHours.pcap_ISCX.csv` | Training data path |
| `SHAP_NSAMPLES` | 100 | SHAP background samples (lower = faster) |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PYTHONUNBUFFERED` | 1 | Disable Python output buffering |

### Dashboard Service (`nids-dashboard`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8501 | Streamlit port (auto-set by Render) |
| `API_URL` | `https://nids-api.onrender.com` | **UPDATE THIS!** |
| `PYTHONUNBUFFERED` | 1 | Disable Python output buffering |

---

## Updating Your Deployment

### Automatic Updates (Recommended)

Render automatically redeploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push origin main

# Render automatically detects push and redeploys (~3-5 min)
```

### Manual Redeploy

1. Go to Render Dashboard
2. Select service
3. Click "Manual Deploy" → "Deploy latest commit"
4. Wait for rebuild

---

## Cost Breakdown

### Free Tier
- **API**: $0/month (with cold starts)
- **Dashboard**: $0/month (with cold starts)
- **Total**: $0/month

### Starter Tier (Recommended for Production)
- **API**: $7/month (always-on, 512MB RAM)
- **Dashboard**: $7/month (always-on, 512MB RAM)
- **Total**: $14/month

### Pro Tier (High-Traffic)
- **API**: $25/month (always-on, 2GB RAM, auto-scaling)
- **Dashboard**: $25/month (always-on, 2GB RAM, auto-scaling)
- **Total**: $50/month

---

## Next Steps

1. ✅ Deploy to Render.com using steps above
2. 🔗 Share your deployed URLs:
   - API: `https://nids-api.onrender.com`
   - Dashboard: `https://nids-dashboard.onrender.com`
3. 📊 Test with real network flow data
4. 📈 Monitor performance and logs
5. 💰 Upgrade to paid tier when ready for production

---

## Support

- **Render Documentation**: https://render.com/docs
- **GitHub Repository**: https://github.com/byessilyurt/network-intrusion-detection
- **Report Issues**: https://github.com/byessilyurt/network-intrusion-detection/issues

---

**Deployment Time**: 5-10 minutes
**Cost**: $0 (Free Tier) or $14/month (Production)
**Status**: Ready to Deploy! 🚀
