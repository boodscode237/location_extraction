HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Location Extractor (spaCy & BiLSTM-CRF)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc; /* bg-slate-50 */
        }
        .loader {
            border: 4px solid #e5e7eb; /* border-gray-200 */
            border-top: 4px solid #3b82f6; /* border-blue-500 */
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 8px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loader.hidden {
             display: none;
        }
        .result-tag {
            display: inline-block;
            background-color: #e0e7ff; /* bg-indigo-100 */
            color: #4338ca; /* text-indigo-700 */
            padding: 4px 10px;
            margin: 4px;
            border-radius: 16px; /* rounded-full */
            font-size: 0.875rem; /* text-sm */
            font-weight: 500; /* medium */
        }
        .toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-size: 0.875rem;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease-in-out, transform 0.3s ease-in-out;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .toast.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        .toast.error {
            background-color: #ef4444; /* bg-red-500 */
        }
        .toast.success {
            background-color: #22c55e; /* bg-green-500 */
        }
        /* Ensure buttons are consistently sized and text centered */
        button {
            min-width: 220px; /* Adjust as needed */
        }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center justify-center p-4 selection:bg-indigo-500 selection:text-white">
    <div class="bg-white p-6 sm:p-8 rounded-xl shadow-xl w-full max-w-2xl">
        <header class="text-center mb-6">
            <h1 class="text-3xl sm:text-4xl font-bold text-gray-800">
                Location <span class="text-indigo-600">Extractor</span>
            </h1>
            <p class="text-gray-600 mt-2 text-sm sm:text-base">
                Identify geographical locations in your text using advanced NLP models.
            </p>
        </header>

        <textarea id="inputText"
                  class="w-full h-36 p-3.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition duration-200 ease-in-out mb-5 resize-y shadow-sm"
                  placeholder="e.g., We travelled from Paris to Berlin, stopping briefly in Brussels. Then we flew to New York."></textarea>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <button id="btnSpacy"
                    onclick="extractLocations('/extract-with-spacy/', 'btnSpacy', 'loaderSpacy')"
                    class="flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-5 rounded-lg transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-60 shadow hover:shadow-md">
                Extract with spaCy
                <span id="loaderSpacy" class="loader hidden"></span>
            </button>
            <button id="btnBiLSTM"
                    onclick="extractLocations('/extract-with-bilstm/', 'btnBiLSTM', 'loaderBiLSTM')"
                    class="flex items-center justify-center bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2.5 px-5 rounded-lg transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-60 shadow hover:shadow-md">
                Extract with BiLSTM-CRF
                <span id="loaderBiLSTM" class="loader hidden"></span>
            </button>
        </div>
        
        <div id="toast-container"></div>

        <div class="mt-3">
            <h2 class="text-lg font-semibold text-gray-700 mb-1">Results:</h2>
            <div id="result"
                 class="p-4 border border-gray-200 bg-slate-50 rounded-lg min-h-[60px] text-gray-700 whitespace-pre-wrap break-words shadow-inner">
                 <span class="text-gray-400">Locations will appear here...</span>
            </div>
            <p id="modelUsed" class="text-xs text-gray-500 text-right mt-1 pr-1"></p>
        </div>
    </div>
    <footer class="text-center mt-8 text-xs text-gray-500">
        <p>&copy; 2024 Location Extractor API. All rights reserved.</p>
    </footer>

    <script>
        const inputTextEl = document.getElementById('inputText');
        const resultDiv = document.getElementById('result');
        const modelUsedP = document.getElementById('modelUsed');
        const btnSpacy = document.getElementById('btnSpacy');
        const btnBiLSTM = document.getElementById('btnBiLSTM');
        const allButtons = [btnSpacy, btnBiLSTM];

        function showToast(message, type = 'error') {
            const toastContainer = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            toastContainer.appendChild(toast);

            // Trigger reflow to enable animation
            toast.offsetHeight; 
            toast.classList.add('show');

            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => {
                    if (toast.parentNode) {
                         toast.parentNode.removeChild(toast);
                    }
                }, 300); // Wait for fade out animation
            }, 3000); // Duration toast is visible
        }

        async function extractLocations(endpoint, buttonId, loaderId) {
            const text = inputTextEl.value.trim();
            if (!text) {
                showToast('Please enter some text to analyze.', 'error');
                inputTextEl.focus();
                return;
            }

            const button = document.getElementById(buttonId);
            const loader = document.getElementById(loaderId);

            // Clear previous results
            resultDiv.innerHTML = '<span class="text-gray-400">Processing...</span>';
            modelUsedP.innerText = '';
            
            allButtons.forEach(btn => btn.disabled = true);
            loader.classList.remove('hidden');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (!response.ok) {
                    const errorMsg = data.detail || `HTTP error! Status: ${response.status}`;
                    throw new Error(errorMsg);
                }
                
                resultDiv.innerHTML = ''; // Clear "Processing..."
                if (data.extracted_locations && data.extracted_locations.length > 0) {
                    data.extracted_locations.forEach(loc => {
                        const tag = document.createElement('span');
                        tag.className = 'result-tag';
                        tag.textContent = loc;
                        resultDiv.appendChild(tag);
                    });
                } else {
                    resultDiv.innerHTML = '<span class="text-gray-500">No locations found.</span>';
                }
                modelUsedP.innerText = `Model: ${data.model_used}`;
                if (data.extracted_locations && data.extracted_locations.length > 0){
                    showToast('Extraction successful!', 'success');
                }

            } catch (error) {
                console.error('Error:', error);
                resultDiv.innerHTML = '<span class="text-red-500">Extraction failed.</span>';
                showToast(error.message || 'An unknown error occurred.', 'error');
            } finally {
                loader.classList.add('hidden');
                allButtons.forEach(btn => btn.disabled = false);
            }
        }
    </script>
</body>
</html>
"""
