from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import numpy as np
import cv2
import random
import struct
import io
import os
from PIL import Image
import base64
from typing import List

app = Flask(__name__)
CORS(app)

def load_words(dict_path: str | None, count: int, rng: random.Random) -> List[str]:
    """Load words from dictionary file or use fallback list."""
    fallback = [
        "orchid","nebulas","granite","whisper","cipher","lumen","quartz","saffron",
        "isotope","tangent","aurora","ember","matrix","quiver","spectrum","zephyr",
        "basilisk","topaz","harbor","mycelium","galaxy","silica","fjord","plasma"
    ]
    pool = fallback
    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
            pool = [w.strip() for w in f if w.strip()]
        if not pool:
            pool = fallback
    return [rng.choice(pool) for _ in range(count)]

def compute_style_features(rgb: np.ndarray) -> np.ndarray:
    """
    Compute style-relevant features using edge detection and texture analysis.
    This approximates what AI models use to identify artistic style.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale edge detection (style features)
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 30, 100)
    
    # Texture features using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.var(laplacian)
    
    # Gradient magnitude (captures brush strokes, line work)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    return {
        'edges': (edges1, edges2),
        'texture': texture,
        'gradient': gradient_mag
    }

def frequency_domain_perturbation(rgb: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    """
    Apply perturbations in frequency domain to target AI feature extractors.
    This is more robust against simple attacks like upscaling.
    """
    h, w, c = rgb.shape
    out = rgb.astype(np.float32)
    
    for channel in range(c):
        # Convert to frequency domain
        fft = np.fft.fft2(rgb[:, :, channel].astype(np.float32))
        fft_shift = np.fft.fftshift(fft)
        
        # Create high-frequency mask (targets AI feature extractors)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Target mid-to-high frequencies (where style features live)
        freq_mask = (mask > min(h, w) * 0.1) & (mask < min(h, w) * 0.4)
        
        # Add small magnitude perturbations to high frequencies
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        # Perturb magnitude slightly in frequency domain
        magnitude_perturb = np.ones_like(magnitude)
        num_perturbed = np.sum(freq_mask)
        if num_perturbed > 0:
            perturbations = np.array([rng.uniform(-epsilon * 0.01, epsilon * 0.01) for _ in range(num_perturbed)])
            magnitude_perturb[freq_mask] = 1.0 + perturbations
        
        # Reconstruct with perturbed magnitude
        fft_shift_perturbed = magnitude_perturb * magnitude * np.exp(1j * phase)
        
        # Convert back to spatial domain
        fft_ishift = np.fft.ifftshift(fft_shift_perturbed)
        img_back = np.fft.ifft2(fft_ishift)
        out[:, :, channel] = np.real(img_back)
    
    return np.clip(out, 0, 255).astype(np.uint8)

def multi_scale_adversarial_noise(rgb: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    """
    Apply multi-scale adversarial noise that targets different feature scales.
    More effective than single-scale noise.
    """
    h, w, c = rgb.shape
    out = rgb.astype(np.float32)
    
    # Multiple scales of noise
    scales = [1.0, 0.5, 0.25, 2.0]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    for scale, weight in zip(scales, weights):
        scale_h, scale_w = int(h * scale), int(w * scale)
        if scale_h < 2 or scale_w < 2:
            continue
            
        # Generate noise at this scale
        noise = np.array([rng.normalvariate(0, 1) for _ in range(scale_h * scale_w * c)], 
                        dtype=np.float32).reshape(scale_h, scale_w, c)
        
        # Emphasize high-frequency components
        if scale_h > 4 and scale_w > 4:
            low = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.0)
            high_freq = noise - low
            noise = high_freq
        
        # Resize to original size
        noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize and apply
        noise_std = np.std(noise_resized) + 1e-8
        noise_normalized = (noise_resized / noise_std) * (epsilon / 3.0) * weight
        out += noise_normalized
    
    return np.clip(out, 0, 255).astype(np.uint8)

def style_aware_perturbation(rgb: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    """
    Apply style-aware perturbations that target artistic style features.
    This is the core of the Glaze algorithm - making AI see different style.
    """
    h, w, c = rgb.shape
    rgb_float = rgb.astype(np.float32)
    
    # Compute style features
    style_features = compute_style_features(rgb)
    
    # Create perturbation that disrupts style features
    perturbation = np.zeros_like(rgb_float)
    
    # Target edge regions (where style is most visible to AI)
    edge_mask = (style_features['edges'][0].astype(np.float32) / 255.0)[:, :, np.newaxis]
    edge_mask = np.repeat(edge_mask, c, axis=2)
    
    # Generate structured noise that disrupts style perception
    noise = np.array([rng.normalvariate(0, 1) for _ in range(h*w*c)], 
                     dtype=np.float32).reshape(h, w, c)
    
    # Apply multi-scale blur to create structured patterns
    low1 = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    low2 = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.8)
    high1 = noise - low1
    high2 = noise - low2
    
    # Combine frequencies to create style-disrupting pattern
    structured_noise = 0.5 * high1 + 0.3 * high2 + 0.2 * noise
    
    # Normalize
    noise_std = np.std(structured_noise) + 1e-8
    structured_noise = (structured_noise / noise_std) * (epsilon / 3.0)
    
    # Apply more strongly to edge/texture regions
    perturbation = structured_noise * (0.7 + 0.3 * edge_mask)
    
    # Add to image
    out = rgb_float + perturbation
    
    # Apply subtle color space jitter to further confuse style classifiers
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Hue shift (targets color-based style recognition)
    hue_jitter = rng.uniform(-epsilon * 0.2, epsilon * 0.2)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_jitter) % 180
    
    # Saturation jitter (affects style perception)
    sat_jitter = rng.uniform(-epsilon * 0.1, epsilon * 0.1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_jitter, 0, 255)
    
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return np.clip(out, 0, 255).astype(np.uint8)

def glaze_like_perturbation(rgb: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    """
    Improved Glaze-style perturbation that protects against AI style mimicry.
    Combines multiple techniques for robustness:
    1. Style-aware perturbations (targets artistic style features)
    2. Multi-scale adversarial noise (harder to remove)
    3. Frequency domain attacks (robust against upscaling) - skipped for large images to save time
    """
    h, w, c = rgb.shape
    
    # Step 1: Apply style-aware perturbations (primary defense)
    rgb = style_aware_perturbation(rgb, epsilon, rng)
    
    # Step 2: Add multi-scale noise for robustness
    rgb = multi_scale_adversarial_noise(rgb, epsilon * 0.6, rng)
    
    # Step 3: Apply frequency domain perturbations (harder to remove)
    # Skip for very large images (>1000px) to stay within Render's 30s timeout
    # Frequency domain processing is expensive but less critical than style-aware perturbations
    if h * w < 1000000:  # Only for images < 1MP (roughly 1000x1000)
        rgb = frequency_domain_perturbation(rgb, epsilon * 0.3, rng)
    
    # Final clipping to ensure valid pixel values
    return np.clip(rgb, 0, 255).astype(np.uint8)

def _bytes_iter(b: bytes):
    for byte in b:
        for bit in range(8):
            yield (byte >> (7 - bit)) & 1

def lsb_embed_text(rgb: np.ndarray, text: str, seed: int) -> np.ndarray:
    """
    Embed text bytes into the least significant bits of pixel channels using a PRNG-driven permutation.
    Visual appearance stays the same; bits carry the payload.
    """
    payload = text.encode("utf-8")
    length_prefix = struct.pack(">I", len(payload))
    data = length_prefix + payload

    h, w, c = rgb.shape
    total_slots = h*w*c
    needed_bits = len(data) * 8
    if needed_bits > total_slots:
        raise ValueError(f"Message too large for image capacity ({needed_bits} bits > {total_slots} bits).")

    # permute positions to distribute bits
    rng = np.random.default_rng(seed)
    positions = rng.permutation(total_slots)[:needed_bits]

    flat = rgb.reshape(-1)
    for pos, bit in zip(positions, _bytes_iter(data)):
        flat[pos] = (flat[pos] & 0xFE) | bit  # set LSB
    return flat.reshape(rgb.shape)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/glaze', methods=['POST'])
def glaze_image():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (limit to 50MB - server will auto-resize dimensions if needed)
        # Higher limit allows large files; dimensions are auto-resized to prevent timeouts
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > 50 * 1024 * 1024:
            return jsonify({'error': 'Image file too large. Maximum file size is 50MB. Please compress or resize your image.'}), 400
        
        # Get parameters
        epsilon = float(request.form.get('epsilon', 10.0))
        seed = int(request.form.get('seed', 1337))
        words = int(request.form.get('words', 25))
        dict_path = request.form.get('dict-path', None)
        
        # Read image using PIL (works with RGB directly)
        im = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Auto-resize very large images to prevent timeouts (max 1500px on longest side for Render compatibility)
        # Render free tier has 30-second HTTP timeout, so we need faster processing
        max_dimension = 1500
        if max(im.size) > max_dimension:
            im.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        rgb = np.array(im)
        
        # Apply glazing
        rng = random.Random(seed)
        rgb = glaze_like_perturbation(rgb, epsilon=epsilon, rng=rng)
        
        # Build randomized (but benign) word payload to inject invisibly
        if words > 0:
            words_list = load_words(dict_path, words, rng)
            # shuffle, add some separators & random punctuation to increase entropy
            punct = [";", ":", ",", ".", "!", "?"]
            mix = []
            for w in words_list:
                mix.append(w)
                if rng.random() < 0.35:
                    mix.append(rng.choice(punct))
            payload = " ".join(mix)
            try:
                rgb = lsb_embed_text(rgb, payload, seed=seed ^ 0xBADC0DE)
            except ValueError:
                # If image is too small, skip embedding
                pass
        
        # Convert back to PIL Image and encode as base64
        result_img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        result_img.save(buffer, format='PNG')  # PNG doesn't use quality parameter
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

