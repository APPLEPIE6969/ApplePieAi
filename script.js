class NeuralNetwork {
    constructor(layers, activation = 'relu') {
        this.layers = layers;
        this.activation = activation;
        this.weights = [];
        this.biases = [];
        this.initializeWeights();
        this.memoryOptimized = true;
    }

    initializeWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const inputSize = this.layers[i];
            const outputSize = this.layers[i + 1];
            
            // Smaller initialization for memory efficiency
            const limit = Math.sqrt(2 / (inputSize + outputSize));
            this.weights.push(this.randomMatrix(outputSize, inputSize, -limit, limit));
            this.biases.push(this.randomMatrix(outputSize, 1, -limit, limit));
        }
    }

    randomMatrix(rows, cols, min = -1, max = 1) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push(Math.random() * (max - min) + min);
            }
            matrix.push(row);
        }
        return matrix;
    }

    activationFunction(x, derivative = false) {
        switch (this.activation) {
            case 'relu':
                if (derivative) {
                    return x.map(val => val > 0 ? 1 : 0);
                }
                return x.map(val => Math.max(0, val));
            
            case 'sigmoid':
                if (derivative) {
                    const sig = this.activationFunction(x);
                    return sig.map(val => val * (1 - val));
                }
                return x.map(val => 1 / (1 + Math.exp(-val)));
            
            case 'tanh':
                if (derivative) {
                    const tanh = this.activationFunction(x);
                    return tanh.map(val => 1 - val * val);
                }
                return x.map(val => Math.tanh(val));
            
            case 'softmax':
                if (derivative) {
                    // Softmax derivative is more complex, simplified here
                    const soft = this.activationFunction(x);
                    return soft.map(val => val * (1 - val));
                }
                const expX = x.map(val => Math.exp(val));
                const sumExp = expX.reduce((a, b) => a + b, 0);
                return expX.map(val => val / sumExp);
            
            default:
                return x;
        }
    }

    matrixMultiply(A, B) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            result[i] = [];
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < B.length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    matrixAdd(A, B) {
        return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    }

    matrixSubtract(A, B) {
        return A.map((row, i) => row.map((val, j) => val - B[i][j]));
    }

    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    forward(input) {
        // Clear previous activations to save memory
        this.activations = [input];
        let current = input;

        for (let i = 0; i < this.weights.length; i++) {
            current = this.matrixMultiply(this.weights[i], current);
            current = this.matrixAdd(current, this.biases[i]);
            
            if (i === this.weights.length - 1 && this.activation === 'softmax') {
                current = this.activationFunction(current);
            } else if (i < this.weights.length - 1) {
                current = this.activationFunction(current);
            } else if (this.activation !== 'softmax') {
                current = this.activationFunction(current);
            }
            
            this.activations.push(current);
        }

        return current;
    }

    backward(X, y, learningRate) {
        const output = this.forward(X);
        const numLayers = this.weights.length;
        
        // Calculate output error
        let error = this.matrixSubtract(output, y);
        
        // Backpropagate
        for (let i = numLayers - 1; i >= 0; i--) {
            const activation = this.activations[i + 1];
            const prevActivation = this.activations[i];
            
            // Calculate gradient
            let gradient;
            if (i === numLayers - 1 && this.activation === 'softmax') {
                gradient = error;
            } else {
                const activationDerivative = this.activationFunction(activation, true);
                gradient = [];
                for (let j = 0; j < error.length; j++) {
                    gradient[j] = [];
                    for (let k = 0; k < error[0].length; k++) {
                        gradient[j][k] = error[j][k] * activationDerivative[j][k];
                    }
                }
            }
            
            // Update weights and biases
            const weightGradient = this.matrixMultiply(gradient, this.transpose(prevActivation));
            this.weights[i] = this.matrixSubtract(this.weights[i], 
                this.scalarMultiply(weightGradient, learningRate));
            this.biases[i] = this.matrixSubtract(this.biases[i], 
                this.scalarMultiply(gradient, learningRate));
            
            // Calculate error for next layer
            if (i > 0) {
                error = this.matrixMultiply(this.transpose(this.weights[i]), error);
            }
        }
    }

    scalarMultiply(matrix, scalar) {
        return matrix.map(row => row.map(val => val * scalar));
    }

    predict(input) {
        const output = this.forward(input);
        return output.map(row => row[0]);
    }

    save() {
        return {
            layers: this.layers,
            activation: this.activation,
            weights: this.weights,
            biases: this.biases
        };
    }

    load(modelData) {
        this.layers = modelData.layers;
        this.activation = modelData.activation;
        this.weights = modelData.weights;
        this.biases = modelData.biases;
    }
}

class Dataset {
    static getMNIST() {
        // Memory-efficient MNIST-like data generation
        const trainingData = [];
        const testData = [];
        
        // Reduced dataset size for 512MB RAM
        for (let i = 0; i < 200; i++) {
            const digit = Math.floor(Math.random() * 10);
            const input = this.randomVector(784);
            const output = new Array(10).fill(0);
            output[digit] = 1;
            
            if (i < 160) {
                trainingData.push({ input: input.map(v => [v]), output: output.map(v => [v]) });
            } else {
                testData.push({ input: input.map(v => [v]), output: output.map(v => [v]) });
            }
        }
        
        return { trainingData, testData };
    }

    static getIris() {
        // Simplified Iris dataset
        const trainingData = [
            { input: [[5.1], [3.5], [1.4], [0.2]], output: [[1], [0], [0]] },
            { input: [[4.9], [3.0], [1.4], [0.2]], output: [[1], [0], [0]] },
            { input: [[7.0], [3.2], [4.7], [1.4]], output: [[0], [1], [0]] },
            { input: [[6.4], [3.2], [4.5], [1.5]], output: [[0], [1], [0]] },
            { input: [[6.3], [3.3], [6.0], [2.5]], output: [[0], [0], [1]] },
            { input: [[5.8], [2.7], [5.1], [1.9]], output: [[0], [0], [1]] }
        ];
        
        return { trainingData, testData: trainingData };
    }

    static getXOR() {
        const trainingData = [
            { input: [[0], [0]], output: [[0]] },
            { input: [[0], [1]], output: [[1]] },
            { input: [[1], [0]], output: [[1]] },
            { input: [[1], [1]], output: [[0]] }
        ];
        
        return { trainingData, testData: trainingData };
    }

    static randomVector(size) {
        const vector = [];
        for (let i = 0; i < size; i++) {
            vector.push(Math.random());
        }
        return vector;
    }
}

class ChartManager {
    constructor() {
        this.lossChart = document.getElementById('lossChart');
        this.accuracyChart = document.getElementById('accuracyChart');
        this.lossCtx = this.lossChart.getContext('2d');
        this.accuracyCtx = this.accuracyChart.getContext('2d');
        this.lossData = [];
        this.accuracyData = [];
    }

    updateCharts(loss, accuracy) {
        this.lossData.push(loss);
        this.accuracyData.push(accuracy);
        
        this.drawChart(this.lossCtx, this.lossData, 'Training Loss', '#e53e3e');
        this.drawChart(this.accuracyCtx, this.accuracyData, 'Accuracy', '#38a169');
    }

    drawChart(ctx, data, label, color) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (data.length === 0) return;
        
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Draw axes
        ctx.strokeStyle = '#cbd5e0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Draw data
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const maxVal = Math.max(...data);
        const minVal = Math.min(...data);
        const range = maxVal - minVal || 1;
        
        data.forEach((value, index) => {
            const x = padding + (index / (data.length - 1 || 1)) * chartWidth;
            const y = height - padding - ((value - minVal) / range) * chartHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw label
        ctx.fillStyle = '#4a5568';
        ctx.font = '12px Arial';
        ctx.fillText(label, padding, padding - 10);
    }

    reset() {
        this.lossData = [];
        this.accuracyData = [];
        this.drawChart(this.lossCtx, [], 'Training Loss', '#e53e3e');
        this.drawChart(this.accuracyCtx, [], 'Accuracy', '#38a169');
    }
}

class AITrainer {
    constructor() {
        this.network = null;
        this.dataset = null;
        this.isTraining = false;
        this.currentEpoch = 0;
        this.trainingStartTime = null;
        this.chartManager = new ChartManager();
        this.progressInterval = null;
        this.initializeEventListeners();
        this.log('AI Trainer initialized', 'info');
    }

    initializeEventListeners() {
        document.getElementById('initializeBtn').addEventListener('click', () => this.initializeNetwork());
        document.getElementById('trainBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopTraining());
        document.getElementById('testBtn').addEventListener('click', () => this.testModel());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveModel());
        document.getElementById('loadBtn').addEventListener('click', () => this.loadModel());
        document.getElementById('addLayerBtn').addEventListener('click', () => this.addHiddenLayer());
        document.getElementById('datasetType').addEventListener('change', () => this.updateDatasetSection());
        document.getElementById('loadDataBtn').addEventListener('click', () => this.loadCustomData());
        document.getElementById('predictBtn').addEventListener('click', () => this.makePrediction());
    }

    addHiddenLayer() {
        const hiddenLayers = document.getElementById('hiddenLayers');
        const newInput = document.createElement('input');
        newInput.type = 'number';
        newInput.className = 'layer-size';
        newInput.value = '32';
        newInput.min = '1';
        newInput.max = '1000';
        hiddenLayers.appendChild(newInput);
    }

    updateDatasetSection() {
        const datasetType = document.getElementById('datasetType').value;
        const customDataSection = document.getElementById('customDataSection');
        customDataSection.style.display = datasetType === 'custom' ? 'block' : 'none';
    }

    loadCustomData() {
        const fileInput = document.getElementById('dataFile');
        const file = fileInput.files[0];
        
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    this.dataset = {
                        trainingData: data.training || [],
                        testData: data.testing || data.training || []
                    };
                    this.log('Custom dataset loaded successfully', 'success');
                } catch (error) {
                    this.log('Error loading custom data: ' + error.message, 'error');
                }
            };
            reader.readAsText(file);
        }
    }

    initializeNetwork() {
        try {
            const inputSize = parseInt(document.getElementById('inputSize').value);
            const outputSize = parseInt(document.getElementById('outputSize').value);
            const hiddenLayerInputs = document.querySelectorAll('.layer-size');
            const activation = document.getElementById('activation').value;
            
            const layers = [inputSize];
            hiddenLayerInputs.forEach(input => {
                const size = parseInt(input.value);
                if (size > 0) layers.push(size);
            });
            layers.push(outputSize);
            
            this.network = new NeuralNetwork(layers, activation);
            this.chartManager.reset();
            this.log(`✅ Network initialized: ${layers.join('-')} layers with ${activation} activation`, 'success');
            
            // Load dataset
            this.loadDataset();
            this.setupTestInputs();
            
            // Enable training button
            document.getElementById('trainBtn').disabled = false;
            
        } catch (error) {
            this.log(`❌ Error initializing network: ${error.message}`, 'error');
        }
    }

    loadDataset() {
        const datasetType = document.getElementById('datasetType').value;
        
        switch (datasetType) {
            case 'mnist':
                this.dataset = Dataset.getMNIST();
                break;
            case 'iris':
                this.dataset = Dataset.getIris();
                break;
            case 'xor':
                this.dataset = Dataset.getXOR();
                break;
            case 'custom':
                if (!this.dataset) {
                    this.log('Please load custom data first', 'warning');
                }
                break;
        }
        
        if (this.dataset) {
            this.log(`Dataset loaded: ${datasetType}`, 'info');
        }
    }

    setupTestInputs() {
        const testInputs = document.getElementById('testInputs');
        testInputs.innerHTML = '';
        
        if (!this.network || !this.dataset) return;
        
        const inputSize = this.network.layers[0];
        const sampleInput = this.dataset.testData[0]?.input || Array(inputSize).fill([0]);
        
        for (let i = 0; i < inputSize; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.placeholder = `Input ${i + 1}`;
            input.value = sampleInput[i]?.[0] || 0;
            input.step = '0.01';
            input.style.width = '100%';
            testInputs.appendChild(input);
        }
    }

    async startTraining() {
        if (!this.network) {
            this.log('❌ Please initialize network first', 'error');
            return;
        }
        
        if (!this.dataset) {
            this.log('❌ Please load dataset first', 'error');
            return;
        }
        
        if (this.isTraining) {
            this.log('⚠️ Training already in progress', 'warning');
            return;
        }
        
        this.isTraining = true;
        this.currentEpoch = 0;
        this.trainingStartTime = Date.now();
        
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        
        this.log(`🚀 Starting training: ${epochs} epochs, LR=${learningRate}, Batch=${batchSize}`, 'info');
        this.log(`📊 Dataset: ${this.dataset.trainingData.length} training samples`, 'info');
        
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        
        // Start real-time progress updates
        this.startProgressMonitoring(epochs);
        
        let bestAccuracy = 0;
        let patienceCounter = 0;
        const maxPatience = 100;
        let currentLearningRate = learningRate;
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            await this.trainEpoch(currentLearningRate, batchSize);
            this.currentEpoch = epoch + 1;
            this.updateUI();
            
            const metrics = this.calculateMetrics();
            
            // Adaptive learning rate
            if (epoch > 100 && metrics.accuracy > 90) {
                currentLearningRate *= 0.999;
            }
            if (epoch > 200 && metrics.accuracy > 95) {
                currentLearningRate *= 0.995;
            }
            
            // Early stopping with patience
            if (metrics.accuracy > bestAccuracy) {
                bestAccuracy = metrics.accuracy;
                patienceCounter = 0;
                this.log(`🎯 New best accuracy: ${bestAccuracy.toFixed(2)}%`, 'success');
            } else {
                patienceCounter++;
            }
            
            // Auto-stop if no improvement
            if (patienceCounter >= maxPatience) {
                this.log(`⏹️ Early stopping - no improvement for ${maxPatience} epochs`, 'warning');
                break;
            }
            
            // Auto-stop if perfect performance
            if (metrics.accuracy >= 99.9) {
                this.log(`🏆 Near-perfect accuracy: ${metrics.accuracy.toFixed(2)}%`, 'success');
                break;
            }
            
            // Memory cleanup
            if (epoch % 10 === 0) {
                this.cleanupMemory();
                await new Promise(resolve => setTimeout(resolve, 50));
            }
        }
        
        this.stopTraining();
        this.log(`✅ Training completed! Best accuracy: ${bestAccuracy.toFixed(2)}%`, 'success');
        
        // Stop progress monitoring
        this.stopProgressMonitoring();
    }

    async trainEpoch(learningRate, batchSize) {
        const { trainingData } = this.dataset;
        
        // Use smaller batches for memory efficiency
        const effectiveBatchSize = Math.min(batchSize, 4);
        
        // Shuffle data
        const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
        
        // Train in smaller batches with multiple passes for better learning
        const passes = learningRate < 0.001 ? 3 : 2; // More passes when learning rate is low
        
        for (let pass = 0; pass < passes; pass++) {
            for (let i = 0; i < shuffled.length; i += effectiveBatchSize) {
                const batch = shuffled.slice(i, i + effectiveBatchSize);
                
                for (const sample of batch) {
                    this.network.backward(sample.input, sample.output, learningRate);
                }
                
                // Clear activations to free memory
                if (i % (effectiveBatchSize * 2) === 0) {
                    this.network.activations = null;
                }
            }
        }
        
        // Calculate metrics
        const metrics = this.calculateMetrics();
        this.chartManager.updateCharts(metrics.loss, metrics.accuracy);
        
        if (this.currentEpoch % 5 === 0) {
            this.log(`Epoch ${this.currentEpoch}: Loss=${metrics.loss.toFixed(4)}, Acc=${metrics.accuracy.toFixed(2)}%`, 'info');
        }
    }

    calculateMetrics() {
        if (!this.dataset || !this.network) {
            return { loss: 0, accuracy: 0 };
        }
        
        let totalLoss = 0;
        let correct = 0;
        const { testData } = this.dataset;
        
        for (const sample of testData) {
            const prediction = this.network.forward(sample.input);
            const loss = this.calculateLoss(prediction, sample.output);
            totalLoss += loss;
            
            // Check if prediction is correct
            const predIndex = this.argmax(prediction.map(p => p[0]));
            const trueIndex = this.argmax(sample.output.map(o => o[0]));
            if (predIndex === trueIndex) {
                correct++;
            }
        }
        
        return {
            loss: totalLoss / testData.length,
            accuracy: (correct / testData.length) * 100
        };
    }

    calculateLoss(prediction, target) {
        // Mean squared error
        let loss = 0;
        for (let i = 0; i < prediction.length; i++) {
            const diff = prediction[i][0] - target[i][0];
            loss += diff * diff;
        }
        return loss / prediction.length;
    }

    argmax(array) {
        let maxIndex = 0;
        let maxValue = array[0];
        for (let i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    stopTraining() {
        this.isTraining = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        
        if (this.trainingStartTime) {
            const duration = Date.now() - this.trainingStartTime;
            const seconds = Math.floor(duration / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            
            this.log(`Training stopped. Duration: ${hours}h ${minutes % 60}m ${seconds % 60}s`, 'info');
        }
        
        // Update progress bar to show completion
        this.updateProgressBar(100, 'Training completed');
    }

    testModel() {
        if (!this.network || !this.dataset) {
            this.log('Please initialize network and load dataset first', 'error');
            return;
        }
        
        const metrics = this.calculateMetrics();
        this.log(`Test Results: Loss=${metrics.loss.toFixed(4)}, Accuracy=${metrics.accuracy.toFixed(2)}%`, 'success');
        
        // Update UI
        document.getElementById('validationLoss').textContent = metrics.loss.toFixed(4);
        document.getElementById('accuracy').textContent = metrics.accuracy.toFixed(2) + '%';
    }

    makePrediction() {
        if (!this.network) {
            this.log('Please initialize network first', 'error');
            return;
        }
        
        const testInputs = document.getElementById('testInputs').querySelectorAll('input');
        const input = [];
        
        testInputs.forEach(inputElement => {
            input.push([parseFloat(inputElement.value) || 0]);
        });
        
        const prediction = this.network.predict(input);
        const resultsDiv = document.getElementById('predictionResults');
        
        let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">';
        
        prediction.forEach((value, index) => {
            const percentage = (value * 100).toFixed(2);
            const barWidth = Math.min(value * 100, 100);
            html += `
                <div style="background: #f7fafc; padding: 10px; border-radius: 5px;">
                    <div>Class ${index}: ${percentage}%</div>
                    <div style="background: #e2e8f0; height: 10px; border-radius: 5px; margin-top: 5px;">
                        <div style="background: #667eea; height: 100%; width: ${barWidth}%; border-radius: 5px;"></div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        resultsDiv.innerHTML = html;
        
        this.log('Prediction completed', 'success');
    }

    saveModel() {
        if (!this.network) {
            this.log('No model to save', 'error');
            return;
        }
        
        const modelData = this.network.save();
        const json = JSON.stringify(modelData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'neural_network_model.json';
        a.click();
        
        URL.revokeObjectURL(url);
        this.log('Model saved successfully', 'success');
    }

    loadModel() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    try {
                        const modelData = JSON.parse(event.target.result);
                        this.network = new NeuralNetwork([1]); // Dummy initialization
                        this.network.load(modelData);
                        this.log('Model loaded successfully', 'success');
                        
                        // Update UI to reflect loaded model
                        this.setupTestInputs();
                    } catch (error) {
                        this.log('Error loading model: ' + error.message, 'error');
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }

    updateUI() {
        document.getElementById('currentEpoch').textContent = this.currentEpoch;
        
        if (this.trainingStartTime) {
            const elapsed = Date.now() - this.trainingStartTime;
            const seconds = Math.floor(elapsed / 1000) % 60;
            const minutes = Math.floor(elapsed / 60000) % 60;
            const hours = Math.floor(elapsed / 3600000);
            
            document.getElementById('trainingTime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        const metrics = this.calculateMetrics();
        document.getElementById('trainingLoss').textContent = metrics.loss.toFixed(4);
        document.getElementById('accuracy').textContent = metrics.accuracy.toFixed(2) + '%';
    }

    startProgressMonitoring(totalEpochs) {
        // Update progress bar every second
        this.progressInterval = setInterval(() => {
            if (this.isTraining && this.currentEpoch > 0) {
                const progress = (this.currentEpoch / totalEpochs) * 100;
                const metrics = this.calculateMetrics();
                
                // Update progress bar
                this.updateProgressBar(progress, `Epoch ${this.currentEpoch}/${totalEpochs} - Accuracy: ${metrics.accuracy.toFixed(1)}%`);
                
                // Log accuracy improvement every second
                if (metrics.accuracy > 0) {
                    console.log(`Current accuracy: ${metrics.accuracy.toFixed(2)}%`);
                }
            }
        }, 1000); // Update every second
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    updateProgressBar(percentage, text) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressFill.style.width = `${Math.min(percentage, 100)}%`;
        progressText.textContent = text;
        
        // Change color based on progress
        if (percentage >= 95) {
            progressFill.style.background = 'linear-gradient(90deg, #38a169 0%, #48bb78 100%)';
        } else if (percentage >= 50) {
            progressFill.style.background = 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)';
        }
    }

    cleanupMemory() {
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
        
        // Clear chart data periodically to save memory
        if (this.chartManager.lossData.length > 100) {
            this.chartManager.lossData = this.chartManager.lossData.slice(-50);
            this.chartManager.accuracyData = this.chartManager.accuracyData.slice(-50);
        }
        
        // Clear network activations
        if (this.network) {
            this.network.activations = null;
        }
    }

    log(message, type = 'info') {
        const logOutput = document.getElementById('logOutput');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
        
        // Keep only last 100 entries
        while (logOutput.children.length > 100) {
            logOutput.removeChild(logOutput.firstChild);
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new AITrainer();
});
