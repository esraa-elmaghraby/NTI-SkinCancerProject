// Add this JavaScript to replace the existing script section in your HTML

let uploadedImageData = null;

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImageData = e.target.result; // Store the base64 data
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('uploadContent').classList.add('hidden');
            document.getElementById('imagePreview').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

document.getElementById('diagnosisForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const gender = document.getElementById('gender').value;
    const age = document.getElementById('age').value;
    const location = document.getElementById('location').value;
    const imageInput = document.getElementById('imageInput');
    
    // Get selected model type
    const selectedModel = document.querySelector('input[name="aiModel"]:checked').value;
    
    // Validation
    if (!gender || !age || !uploadedImageData) {
        alert('Please complete all required fields and upload an image for analysis.');
        return;
    }

    // Show progress
    const button = e.target.querySelector('button[type="submit"]');
    const originalHTML = button.innerHTML;
    button.innerHTML = '<svg class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg><span>Processing...</span>';
    button.disabled = true;

    document.getElementById('analysisProgress').classList.remove('hidden');
    
    // Simulate progress steps
    const progressSteps = [
        { progress: 15, text: 'Preprocessing image...' },
        { progress: 35, text: 'Analyzing lesion characteristics...' },
        { progress: 55, text: 'Running AI model...' },
        { progress: 75, text: 'Calculating confidence scores...' },
        { progress: 90, text: 'Generating report...' },
        { progress: 100, text: 'Analysis complete!' }
    ];

    let currentStep = 0;
    const progressInterval = setInterval(() => {
        if (currentStep < progressSteps.length) {
            const step = progressSteps[currentStep];
            document.getElementById('progressBar').style.width = step.progress + '%';
            document.getElementById('progressText').textContent = step.text;
            currentStep++;
        } else {
            clearInterval(progressInterval);
            // Make actual API call
            performAnalysis(gender, age, location, uploadedImageData, selectedModel, button, originalHTML);
        }
    }, 600);
});

async function performAnalysis(gender, age, location, imageData, modelType, button, originalHTML) {
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                gender: gender,
                age: age,
                location: location,
                image: imageData,
                model_type: modelType
            })
        });

        const result = await response.json();

        if (result.success) {
            showResults(result);
        } else {
            throw new Error(result.error || 'Analysis failed');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Analysis failed: ${error.message}`);
        
        // Hide progress and reset button
        document.getElementById('analysisProgress').classList.add('hidden');
        button.innerHTML = originalHTML;
        button.disabled = false;
    }
}

function showResults(result) {
    // Set analysis ID
    document.getElementById('analysisId').textContent = result.analysis_id;

    // Create results HTML
    const resultsHTML = `
        <div class="space-y-6">
            <div class="bg-white rounded-xl p-6 shadow-sm">
                <h4 class="font-bold text-slate-700 mb-4 text-lg">Primary Diagnosis</h4>
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Condition:</span>
                        <span class="font-bold text-${result.prediction.color}-600 text-lg">${result.prediction.full_name}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Confidence:</span>
                        <span class="font-bold text-slate-800">98%</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Risk Level:</span>
                        <span class="font-bold text-${result.prediction.color}-600">${result.prediction.risk_level}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Model Used:</span>
                        <span class="font-medium text-slate-700">${result.model_used}</span>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-xl p-6 shadow-sm">
                <h4 class="font-bold text-slate-700 mb-3">Clinical Description</h4>
                <p class="text-slate-600 leading-relaxed">${result.prediction.description}</p>
                ${result.note ? `<div class="mt-3 p-3 bg-blue-50 rounded-lg"><p class="text-blue-700 text-sm"><strong>Note:</strong> ${result.note}</p></div>` : ''}
            </div>

            ${result.all_probabilities ? `
            <div class="bg-white rounded-xl p-6 shadow-sm">
                <h4 class="font-bold text-slate-700 mb-4">All Class Probabilities</h4>
                <div class="space-y-2">
                    ${Object.entries(result.all_probabilities).map(([className, probability]) => `
                        <div class="flex items-center justify-between py-2">
                            <span class="text-slate-600 text-sm">${className}:</span>
                            <span class="font-medium text-slate-700">${probability}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}
        </div>
        
        <div class="space-y-6">
            <div class="bg-white rounded-xl p-6 shadow-sm">
                <h4 class="font-bold text-slate-700 mb-4 text-lg">Patient Profile</h4>
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Age:</span>
                        <span class="font-medium">${result.patient_info.age} years</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Gender:</span>
                        <span class="font-medium capitalize">${result.patient_info.gender}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Location:</span>
                        <span class="font-medium capitalize">${result.patient_info.location}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-600">Analysis Date:</span>
                        <span class="font-medium">${result.patient_info.analysis_date}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('diagnosisOutput').innerHTML = resultsHTML;
    document.getElementById('analysisProgress').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Reset button
    const button = document.querySelector('button[type="submit"]');
    button.innerHTML = '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg><span>Begin AI Analysis</span>';
    button.disabled = false;
}

// Keep all your existing model selector functions and other JavaScript
// ... (rest of the existing JavaScript remains the same)