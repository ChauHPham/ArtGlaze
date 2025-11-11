# Image Glaze - AI Protection Website

A web application that protects your images from AI training by applying imperceptible perturbations (glazing) to make them harder for AI models to learn from.

## Features

- 🖼️ Drag-and-drop image upload
- 🛡️ Apply glazing protection to images
- ⚙️ Adjustable protection strength (epsilon)
- 📥 Download glazed images
- 🎨 Modern, responsive UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload an image**: Drag and drop an image onto the upload area, or click to browse
2. **Adjust settings** (optional):
   - **Protection Strength (ε)**: Higher values = stronger protection (default: 10.0, recommended)
   - **Random Seed**: Controls the randomized pattern of the glaze. Using the same seed with the same settings makes results reproducible (default: 1337).
   - **Payload Words**: Embeds invisible random words in pixel least-significant-bits to add entropy and reduce training utility. Set to 0 to disable (default: 25).
3. **Click "Glaze Image"** to process
4. **Download** the glazed image

## How It Works

The improved Glaze-style algorithm uses multiple defense layers:

### 1. Style-Aware Perturbations (Primary Defense)
- **Targets artistic style features**: Uses edge detection and texture analysis to identify where AI models perceive artistic style
- **Disrupts style perception**: Applies structured noise specifically to edge/texture regions where style is most visible to AI
- **Color space jitter**: Subtle hue and saturation shifts that confuse color-based style recognition

### 2. Multi-Scale Adversarial Noise
- **Multiple frequency scales**: Applies noise at different scales (1x, 0.5x, 0.25x, 2x) to target various feature extractors
- **High-frequency emphasis**: Focuses on frequencies that AI models use for feature extraction
- **Harder to remove**: Multi-scale approach makes simple filtering/upscaling attacks less effective

### 3. Frequency Domain Attacks
- **FFT-based perturbations**: Modifies image in frequency domain (harder to detect/remove)
- **Targets mid-to-high frequencies**: Where style features are encoded in AI models
- **Robust against upscaling**: Frequency domain attacks survive common image processing operations

### 4. LSB Steganography (Optional)
- **Invisible payload embedding**: Embeds random words/punctuation in least significant bits
- **Increases entropy**: Makes training data noisier and less useful for AI models
- **Undetectable**: Completely invisible to human eyes

### Protection Mechanism
The algorithm makes images appear **unchanged to humans** but **dramatically different in style to AI models**. When an AI model tries to learn from a glazed image, it learns incorrect style associations, making style mimicry fail.

## Analysis Utility

Use `analysis.py` to compare an original image, its glazed counterpart, and an AI result:

```bash
python analysis.py \
  --original path/to/original.png \
  --glazed path/to/glazed.png \
  --ai-result path/to/ai_output.png \
  --hist-output histograms.png \
  --stats-output stats.csv \
  --stats-image stats.png \
  --heatmap-output heatmap.png
```

The script prints color statistics, computes a pixel-noise metric, renders RGB histograms (displayed or saved if `--hist-output` is provided), outputs a table image when `--stats-image` is set, and can create a perturbation heatmap overlay via `--heatmap-output`. Use `--heatmap-input` to select a specific image for the heatmap (defaults to the glazed image).

Sample directories are available under `samples/`:
- `samples/original/` — place your original, unprotected artwork
- `samples/glazed/` — store the protected/glazed output
- `samples/ai_result/` — drop any AI-generated comparisons

You can point the analysis script at files in these folders to keep a consistent workflow.

## Technical Details

- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Algorithm**: Adapted from video_glaze.py for single image processing

## Protection Strength and Processing Time

- The protection strength slider controls `ε` (epsilon). Higher values increase the perturbation magnitude and robustness, but do not materially change runtime in this implementation.
- Runtime primarily depends on image resolution and hardware, not `ε`. The heaviest steps are Gaussian blurs, multi-scale resizes, and FFTs, which scale roughly with the number of pixels (FFTs scale as N·log N).

Typical CPU runtimes (single image, glaze step only; excludes upload/download), on a modern laptop CPU:
- ~1024×1024: ~0.2–0.6s
- ~2048×2048: ~0.6–1.2s
- ~4096×4096: ~1.5–3.0s

Effect of `ε`:
- `ε = 7–12` (default 10, recommended): Strong protection with minimal visible change on most art; runtime ~same.
- `ε = 13–16`: Very strong protection; slight texture possible on flat areas; runtime ~same.
- `ε = 17–20`: Aggressive protection; may introduce subtle artifacts; runtime ~same.

Notes:
- GPU acceleration is not used by default. For very large images, consider downscaling before glazing and then upscaling with a high-quality resampler, or integrating GPU-accelerated FFTs/filters if needed.

## Data Storage and Disclaimer

- This demo does not use a database and does not persist uploads. Images are processed in memory during the request and the glazed result is returned to the browser as a base64 data URL. No server-side copies are stored by default.
- While glazing can significantly reduce the usefulness of images for AI style training, complete protection cannot be guaranteed. Adversaries and model pipelines evolve, and effectiveness may vary across models and post-processing steps. Use layered defenses and stay updated with new tooling.

## Export Recommendations

- Preferred protection upload: PNG, 8‑bit RGB, keep metadata, no optimization (larger files compress worse and decode slower).
- Resolution: 2048–4096 px max side for stronger effect; higher resolutions increase file size and decode time.
- Strength: consider ε = 12–14 for extra robustness if slight artifacts are acceptable.
- Sharing-friendly: if you need smaller files while keeping protection, export JPEG (quality 90–95) and cap long edge ~2048 px.

## Deploying to the Web (Quick)

You can run this in production with any Python host. Minimal options:

- Gunicorn locally:
  ```bash
  pip install -r requirements.txt
  gunicorn wsgi:app --bind 0.0.0.0:5000
  ```
- Heroku/Railway/Render:
  - Repo contains `Procfile`, `requirements.txt`, and `wsgi.py`
  - Create a new app, deploy from this folder; default process uses `web: gunicorn wsgi:app`
  - For Render, use `render.yaml` in this repo; it sets a longer Gunicorn timeout suited for image processing workloads. You can override via env vars (`WEB_TIMEOUT`, `WEB_CONCURRENCY`, etc.) based on your plan.

Expose port 5000 (or `${PORT}` provided by the platform). No database is required.

# ArtGlaze
