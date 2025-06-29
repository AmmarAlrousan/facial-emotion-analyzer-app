document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const analyzeButton = document.getElementById('analyzeButton');
    const useWebcamButton = document.getElementById('useWebcamButton');
    const webcamSection = document.querySelector('.webcam-section');
    const webcamVideo = document.getElementById('webcamVideo');
    const captureButton = document.getElementById('captureButton');
    const previewImage = document.getElementById('previewImage');
    const emotionResult = document.getElementById('emotionResult');

    let stream = null; // To store webcam stream

    // ====== Webcam Functions ======
    useWebcamButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = stream;
            webcamSection.style.display = 'block'; // Show webcam section
            previewImage.style.display = 'none'; // Hide uploaded image preview
            emotionResult.textContent = 'Webcam active. Capture an image.';
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            emotionResult.textContent = 'Error accessing webcam. Please allow camera access.';
        }
    });

    captureButton.addEventListener('click', () => {
        if (!stream) {
            emotionResult.textContent = 'Webcam not active. Click "Use Webcam" first.';
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL('image/jpeg'); // Convert to Base64 JPEG
        analyzeImage(imageData);

        // Optionally, stop webcam stream after capture:
        // if (stream) {
        //     stream.getTracks().forEach(track => track.stop());
        //     webcamVideo.srcObject = null;
        //     webcamSection.style.display = 'none';
        // }
    });

    // ====== Image Upload Functions ======
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                webcamSection.style.display = 'none'; // Hide webcam section
                if (stream) { // Stop webcam if active
                    stream.getTracks().forEach(track => track.stop());
                    webcamVideo.srcObject = null;
                }
                emotionResult.textContent = 'Image ready for analysis.';
            };
            reader.readAsDataURL(file); // Read file as Base64
        }
    });

    analyzeButton.addEventListener('click', () => {
        if (previewImage.src && previewImage.src !== '#') {
            analyzeImage(previewImage.src);
        } else {
            emotionResult.textContent = 'Please upload an image or capture from webcam first.';
        }
    });

    // ====== Function to Send Image to Server ======
    async function analyzeImage(imageData) {
        emotionResult.textContent = 'Analyzing...';
        emotionResult.style.color = '#ffc107'; // Yellow for loading

        try {
            const response = await fetch('/analyze_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (response.ok) {
                emotionResult.textContent = `Predicted Emotion: ${data.emotion}`;
                emotionResult.style.color = '#28a745'; // Green for success
            } else {
                emotionResult.textContent = `Error: ${data.error || 'Unknown error'}`;
                emotionResult.style.color = '#dc3545'; // Red for error
                console.error('Server error:', data.error);
            }
        } catch (error) {
            emotionResult.textContent = `Network error: ${error.message}`;
            emotionResult.style.color = '#dc3545';
            console.error('Fetch error:', error);
        }
    }
});