const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const originalImage = document.getElementById('originalImage');
const glazedImage = document.getElementById('glazedImage');
const downloadBtn = document.getElementById('downloadBtn');
const epsilonSlider = document.getElementById('epsilon');
const epsilonValue = document.getElementById('epsilonValue');
const seedInput = document.getElementById('seed');
const wordsInput = document.getElementById('words');

let currentFile = null;
let glazedImageData = null;

// Update epsilon value display
epsilonSlider.addEventListener('input', (e) => {
    epsilonValue.textContent = e.target.value;
});

// Drop zone click
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    currentFile = file;
    processBtn.disabled = false;
    
    // Preview original image
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        resultsSection.style.display = 'block';
        // Reset glazed preview state until processing completes
        glazedImage.src = '';
        glazedImage.style.display = 'none';
        glazedImageData = null;
        downloadBtn.style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    hideError();
}

async function processImage() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }
    
    // Check file size (limit to 10MB)
    if (currentFile.size > 10 * 1024 * 1024) {
        showError('Image too large. Please use images under 10MB. Large images may timeout.');
        return;
    }
    
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    hideError();
    processBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('epsilon', epsilonSlider.value);
    formData.append('seed', seedInput.value);
    formData.append('words', wordsInput.value);
    
    try {
        const response = await fetch('/glaze', {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(180000) // 3 minute (180 second) timeout
        });
        
        // Check if response is OK before trying to parse JSON
        if (!response.ok) {
            let errorMsg = 'Server error occurred';
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch {
                // If response isn't JSON (like 502 HTML error page)
                if (response.status === 502 || response.status === 503) {
                    errorMsg = 'Server timeout - image may be too large or processing took too long. Try a smaller image or lower protection strength.';
                } else if (response.status === 504) {
                    errorMsg = 'Request timeout - processing took too long. Try a smaller image.';
                } else {
                    errorMsg = `Server error (${response.status}). Please try again.`;
                }
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        
        glazedImageData = data.image;
        glazedImage.src = data.image;
        glazedImage.style.display = 'block';
        downloadBtn.style.display = 'inline-block';
        resultsSection.style.display = 'block';
        
    } catch (err) {
        if (err.name === 'AbortError' || err.name === 'TimeoutError') {
            showError('Request timed out. The image may be too large. Try a smaller image or lower protection strength.');
        } else {
            showError(err.message || 'An error occurred while processing the image');
        }
    } finally {
        loading.style.display = 'none';
        processBtn.disabled = false;
    }
}

function downloadImage() {
    if (!glazedImageData) {
        showError('No glazed image available to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = glazedImageData;
    link.download = `glazed_${currentFile.name.replace(/\.[^/.]+$/, '')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

processBtn.addEventListener('click', processImage);
downloadBtn.addEventListener('click', downloadImage);

