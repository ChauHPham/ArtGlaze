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
const seedError = document.getElementById('seedError');
const wordsError = document.getElementById('wordsError');

let currentFile = null;
let glazedImageData = null;

// Update epsilon value display
epsilonSlider.addEventListener('input', (e) => {
    epsilonValue.textContent = e.target.value;
});

// Sanitize seed input function
function sanitizeSeed(value) {
    // Remove all non-numeric characters
    value = value.replace(/[^0-9]/g, '');
    // Limit to 4 digits
    if (value.length > 4) {
        value = value.slice(0, 4);
    }
    return value;
}

// Sanitize and validate seed input (must be exactly 4 digits, 1000-2000)
seedInput.addEventListener('input', (e) => {
    let value = sanitizeSeed(e.target.value);
    e.target.value = value;
    
    // Validate range
    if (value.length === 0) {
        seedError.style.display = 'none';
        seedInput.classList.remove('error');
        return;
    }
    
    const numValue = parseInt(value, 10);
    if (isNaN(numValue) || value.length !== 4 || numValue < 1000 || numValue > 2000) {
        seedError.textContent = 'Random seed must be exactly 4 digits between 1000-2000';
        seedError.style.display = 'block';
        seedInput.classList.add('error');
    } else {
        seedError.style.display = 'none';
        seedInput.classList.remove('error');
    }
});

// Sanitize pasted content for seed
seedInput.addEventListener('paste', (e) => {
    e.preventDefault();
    const pasted = (e.clipboardData || window.clipboardData).getData('text');
    const sanitized = sanitizeSeed(pasted);
    seedInput.value = sanitized;
    seedInput.dispatchEvent(new Event('input'));
});

// Sanitize words input function
function sanitizeWords(value) {
    // Remove all non-numeric characters
    value = value.replace(/[^0-9]/g, '');
    // Limit to 3 digits
    if (value.length > 3) {
        value = value.slice(0, 3);
    }
    return value;
}

// Sanitize and validate words input (must be 1-3 digits, 0-100)
wordsInput.addEventListener('input', (e) => {
    let value = sanitizeWords(e.target.value);
    e.target.value = value;
    
    // Validate range
    if (value.length === 0) {
        wordsError.style.display = 'none';
        wordsInput.classList.remove('error');
        return;
    }
    
    const numValue = parseInt(value, 10);
    if (isNaN(numValue) || numValue < 0 || numValue > 100) {
        wordsError.textContent = 'Payload words must be a number between 0-100';
        wordsError.style.display = 'block';
        wordsInput.classList.add('error');
    } else {
        wordsError.style.display = 'none';
        wordsInput.classList.remove('error');
    }
});

// Sanitize pasted content for words
wordsInput.addEventListener('paste', (e) => {
    e.preventDefault();
    const pasted = (e.clipboardData || window.clipboardData).getData('text');
    const sanitized = sanitizeWords(pasted);
    wordsInput.value = sanitized;
    wordsInput.dispatchEvent(new Event('input'));
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
    
    // Validate inputs before processing
    const seedValue = seedInput.value.trim();
    const wordsValue = wordsInput.value.trim();
    
    // Validate seed
    if (seedValue.length === 0) {
        seedError.textContent = 'Random seed is required';
        seedError.style.display = 'block';
        seedInput.classList.add('error');
        return;
    }
    const seedNum = parseInt(seedValue, 10);
    if (isNaN(seedNum) || seedValue.length !== 4 || seedNum < 1000 || seedNum > 2000) {
        seedError.textContent = 'Random seed must be exactly 4 digits between 1000-2000';
        seedError.style.display = 'block';
        seedInput.classList.add('error');
        return;
    }
    
    // Validate words
    if (wordsValue.length === 0) {
        wordsError.textContent = 'Payload words is required';
        wordsError.style.display = 'block';
        wordsInput.classList.add('error');
        return;
    }
    const wordsNum = parseInt(wordsValue, 10);
    if (isNaN(wordsNum) || wordsNum < 0 || wordsNum > 100) {
        wordsError.textContent = 'Payload words must be a number between 0-100';
        wordsError.style.display = 'block';
        wordsInput.classList.add('error');
        return;
    }
    
    // Clear any previous errors
    seedError.style.display = 'none';
    wordsError.style.display = 'none';
    seedInput.classList.remove('error');
    wordsInput.classList.remove('error');
    
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    hideError();
    processBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('epsilon', epsilonSlider.value);
    // Sanitize: ensure numeric values only
    formData.append('seed', seedNum.toString());
    formData.append('words', wordsNum.toString());
    
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

