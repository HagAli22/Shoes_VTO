# Stage A 13-KP Web Demo for Mobile Testing

Real-time web-based demo for testing the Stage A 13-keypoint foot detection model on mobile devices using camera access and public tunneling.

## Features

- 📱 **Mobile-friendly interface** with camera access
- 🎥 **Real-time inference** from device camera
- 📸 **Image upload** support (photos from gallery)
- 🌐 **Public URL tunneling** via ngrok or Cloudflare
- 📊 **Live performance metrics** (FPS, latency)
- 🎯 **Visual detection overlay** with keypoints and skeleton
- ⚡ **Quick benchmark** functionality

## Quick Start

### Option 1: Double-Click Launcher (Easiest)

1. **Install tunnel tool** (one-time setup):
   ```powershell
   # Option A: ngrok (recommended)
   winget install ngrok
   
   # Option B: cloudflared
   winget install Cloudflare.cloudflared
   ```

2. **Double-click**: `START_MOBILE_DEMO.bat`

3. **Copy the tunnel URL** shown in the terminal

4. **Open URL on mobile** device to test

### Option 2: PowerShell Script

```powershell
cd "F:\Machine learning\Shoes_VTO"

# Start with automatic tunnel detection
.\deliverables\stage_a_13kp\web_demo\start_demo.ps1

# Or specify tunnel type
.\deliverables\stage_a_13kp\web_demo\start_demo.ps1 -Tunnel ngrok
.\deliverables\stage_a_13kp\web_demo\start_demo.ps1 -Tunnel cloudflared

# Or run without tunnel (local network only)
.\deliverables\stage_a_13kp\web_demo\start_demo.ps1 -Tunnel none
```

### Option 3: Manual Python

```bash
# Activate conda environment
conda activate yolo

# Install dependencies
pip install -r deliverables/stage_a_13kp/web_demo/requirements.txt

# Start Flask server
python deliverables/stage_a_13kp/web_demo/app.py

# In another terminal, start tunnel
ngrok http 5000
# OR
cloudflared tunnel --url http://localhost:5000
```

## Architecture

```
web_demo/
├── app.py                      # Flask backend (inference API)
├── templates/
│   └── index.html              # Frontend (camera + UI)
├── requirements.txt            # Python dependencies
├── start_demo.ps1              # PowerShell launcher with tunneling
├── START_MOBILE_DEMO.bat       # Windows batch launcher
└── README.md                   # This file
```

## API Endpoints

### `GET /`
Main web interface

### `GET /api/model_info`
Returns model metadata:
```json
{
  "model_path": "stage-a-320-13kp-v7.onnx",
  "model_size_mb": 12.39,
  "input_name": "images",
  "input_shape": [1, 3, 320, 320],
  "output_name": "output0",
  "output_shape": [1, 45, 2100],
  "providers": ["CPUExecutionProvider"]
}
```

### `POST /api/inference`
Run inference on image:

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [100, 200, 300, 500],
      "class_id": 0,
      "class_name": "left_foot",
      "confidence": 0.95,
      "keypoints": [
        {
          "index": 0,
          "name": "toe_ground",
          "x": 150,
          "y": 250,
          "visibility": 0.98
        },
        // ... 12 more keypoints
      ]
    }
  ],
  "inference_time_ms": 45.2,
  "fps": 22.1,
  "result_image": "data:image/jpeg;base64,..."
}
```

### `GET /api/benchmark`
Run performance benchmark (50 iterations):

**Response:**
```json
{
  "mean_latency_ms": 42.5,
  "median_latency_ms": 41.8,
  "p95_latency_ms": 48.3,
  "mean_fps": 23.5
}
```

## Tunnel Options

### ngrok (Recommended)

**Pros:**
- ✅ Easy to install and use
- ✅ Web UI at http://localhost:4040
- ✅ Shows request history
- ✅ Free tier available

**Installation:**
```powershell
winget install ngrok
# OR download from: https://ngrok.com/download
```

**Usage:**
```bash
ngrok http 5000
```

**Output:**
```
Forwarding  https://abc123.ngrok.io -> http://localhost:5000
```

### Cloudflare Tunnel

**Pros:**
- ✅ No account required for quick tunnels
- ✅ Cloudflare's global network
- ✅ Good performance

**Installation:**
```powershell
winget install Cloudflare.cloudflared
# OR download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
```

**Usage:**
```bash
cloudflared tunnel --url http://localhost:5000
```

**Output:**
```
Your quick Tunnel has been created! Visit it at:
https://xyz789.trycloudflare.com
```

## Mobile Testing Workflow

1. **Start the demo** with tunnel enabled
   ```powershell
   .\deliverables\stage_a_13kp\web_demo\start_demo.ps1
   ```

2. **Copy the public URL** from terminal output
   ```
   📱 MOBILE ACCESS URL:
   https://abc123.ngrok.io
   ```

3. **Open URL on mobile device**
   - Use Safari (iOS) or Chrome (Android)
   - Grant camera permissions when prompted

4. **Test the model**
   - Click "📷 Start Camera"
   - Point camera at feet/shoes
   - Watch real-time detection

5. **Check performance**
   - Monitor FPS and latency in the stats panel
   - Click "⚡ Benchmark" for detailed metrics

## Performance Expectations

### Desktop Testing (Local)
- **Latency**: 20-50ms
- **FPS**: 20-50 FPS
- **CPU Usage**: 50-80% on single core

### Mobile Testing (via Tunnel)
- **Latency**: 100-300ms (includes network overhead)
- **FPS**: 10-20 FPS (limited by network round-trip)
- **Note**: Network adds ~50-150ms overhead

### Actual Device Performance
For production deployment, run inference locally on device:
- **WebGPU (mobile)**: 15-30 FPS
- **WASM (mobile)**: 8-15 FPS

## Troubleshooting

### Camera Access Denied

**Issue**: Browser doesn't prompt for camera access

**Solutions:**
- Ensure HTTPS (tunnels provide this automatically)
- Check browser permissions: Settings → Privacy → Camera
- Try different browser (Chrome recommended)

### Tunnel Not Working

**Issue**: Cannot connect to tunnel URL

**Solutions:**
```powershell
# Check if ngrok is running
curl http://localhost:4040/api/tunnels

# Restart tunnel
# Press Ctrl+C, then run start script again

# Try alternative tunnel
.\start_demo.ps1 -Tunnel cloudflared
```

### Slow Performance

**Issue**: Low FPS or high latency

**Solutions:**
- Close other applications
- Use quantized model (INT8):
  ```powershell
  .\start_demo.ps1 -Model "deliverables/stage_a_13kp/models/quantized/stage-a-320-13kp-v7-int8.onnx"
  ```
- Test on local network first (no tunnel)
- Check CPU usage in Task Manager

### Model Not Found

**Issue**: Error loading model

**Solutions:**
```powershell
# Verify model exists
ls deliverables\stage_a_13kp\models\stage-a-320-13kp-v7.onnx

# Specify model path explicitly
.\start_demo.ps1 -Model "path/to/your/model.onnx"
```

### Port Already in Use

**Issue**: Port 5000 is already occupied

**Solutions:**
```powershell
# Use different port
.\start_demo.ps1 -Port 5001

# Find process using port 5000
netstat -ano | findstr :5000
# Kill process by PID
taskkill /PID <PID> /F
```

## Browser Compatibility

| Browser | Desktop | Mobile | Camera | Notes |
|---------|---------|--------|--------|-------|
| **Chrome** | ✅ | ✅ | ✅ | Best performance |
| **Edge** | ✅ | ✅ | ✅ | Chromium-based |
| **Firefox** | ✅ | ✅ | ✅ | Good compatibility |
| **Safari** | ✅ | ✅ | ⚠️ | Requires HTTPS |
| **Opera** | ✅ | ✅ | ✅ | Chromium-based |

**Requirements:**
- HTTPS (provided by tunnels)
- Camera API support
- Canvas API support
- ES6+ JavaScript

## Security Notes

### Tunnel Security
- ⚠️ Tunnel URLs are **public** by default
- 🔒 Anyone with the URL can access the demo
- ⏰ ngrok free URLs expire after session ends
- 🌐 Cloudflare quick tunnels are temporary

### Best Practices
- Don't share tunnel URLs publicly
- Close tunnel when testing is complete
- Use authentication for production deployments
- Monitor tunnel traffic (ngrok web UI)

### Production Deployment
For production, use:
- Proper SSL certificates
- Authentication/authorization
- Rate limiting
- CORS restrictions
- CDN for model delivery

## Advanced Configuration

### Custom Model

Test different quantized versions:

```powershell
# FP16 model
.\start_demo.ps1 -Model "deliverables/stage_a_13kp/models/quantized/stage-a-320-13kp-v7-fp16.onnx"

# INT8 model (smallest, fastest on CPU)
.\start_demo.ps1 -Model "deliverables/stage_a_13kp/models/quantized/stage-a-320-13kp-v7-int8.onnx"
```

### Environment Variables

```powershell
# Set ONNX Runtime provider
$env:ORT_EXECUTION_PROVIDER="cuda"
python app.py

# Enable GPU if available
$env:CUDA_VISIBLE_DEVICES="0"
python app.py
```

### Flask Debug Mode

```bash
python app.py --debug
```

## Integration Examples

### Embed in Existing Web App

```html
<iframe 
  src="https://your-tunnel-url.ngrok.io" 
  width="100%" 
  height="800px"
  allow="camera"
></iframe>
```

### JavaScript API Client

```javascript
// Take photo from camera
async function captureAndInfer(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  
  const imageData = canvas.toDataURL('image/jpeg', 0.8);
  
  const response = await fetch('/api/inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  });
  
  return await response.json();
}
```

## Next Steps

After mobile testing:

1. **Collect Performance Data**
   - Test on multiple devices
   - Record FPS and latency
   - Document device specifications

2. **Optimize if Needed**
   - Try INT8 quantization for better mobile performance
   - Adjust inference frequency (skip frames)
   - Consider client-side inference (ONNX Runtime Web)

3. **Deploy to Production**
   - Set up proper hosting
   - Implement authentication
   - Add analytics
   - Optimize for scale

## Support

- Check Flask logs for errors
- Monitor ngrok dashboard: http://localhost:4040
- Test locally first before using tunnel
- Verify conda environment is activated

## License

Part of Shoes VTO AI project.
