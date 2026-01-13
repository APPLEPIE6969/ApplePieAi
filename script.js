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

    activationDerivative(x) {
        // Handle both 1D and 2D arrays
        if (Array.isArray(x[0])) {
            // 2D array
            return x.map(row => this.activationDerivative(row));
        }
        
        // 1D array
        switch (this.activation) {
            case 'relu':
                return x.map(val => val > 0 ? 1 : 0);
            case 'sigmoid':
                const sig = this.activationFunction(x);
                return sig.map(val => val * (1 - val));
            case 'tanh':
                const tanh = this.activationFunction(x);
                return tanh.map(val => 1 - val * val);
            case 'softmax':
                const soft = this.activationFunction(x);
                return soft.map(val => val * (1 - val));
            default:
                return x.map(() => 1);
        }
    }

    activationFunction(x) {
        // Handle both 1D and 2D arrays
        if (Array.isArray(x[0])) {
            // 2D array
            return x.map(row => this.activationFunction(row));
        }
        
        // 1D array
        switch (this.activation) {
            case 'relu':
                return x.map(val => Math.max(0, val));
            case 'sigmoid':
                return x.map(val => 1 / (1 + Math.exp(-val)));
            case 'tanh':
                return x.map(val => Math.tanh(val));
            case 'softmax':
                const expX = x.map(val => Math.exp(val));
                const sumExp = expX.reduce((a, b) => a + b, 0);
                return expX.map(val => val / sumExp);
            default:
                return x;
        }
    }

    matrixMultiply(A, B) {
        // Handle different dimensions
        if (!Array.isArray(A[0])) {
            // A is 1D, convert to 2D
            A = [A];
        }
        if (!Array.isArray(B[0])) {
            // B is 1D, convert to 2D
            B = B.map(val => [val]);
        }
        
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
        // Handle both 1D and 2D arrays
        if (A[0] && Array.isArray(A[0])) {
            // 2D arrays
            return A.map((row, i) => row.map((val, j) => val + B[i][j]));
        } else {
            // 1D arrays
            return A.map((val, i) => val + B[i]);
        }
    }

    matrixSubtract(A, B) {
        // Handle both 1D and 2D arrays
        if (A[0] && Array.isArray(A[0])) {
            // 2D arrays
            return A.map((row, i) => row.map((val, j) => val - B[i][j]));
        } else {
            // 1D arrays
            return A.map((val, i) => val - B[i]);
        }
    }

    transpose(matrix) {
        // Handle both 1D and 2D arrays
        if (!Array.isArray(matrix[0])) {
            // 1D array, convert to 2D then transpose
            return [[matrix[0]]];
        }
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
        try {
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
                    const activationDerivative = this.activationDerivative(activation);
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
        } catch (error) {
            console.error('Error in backward pass:', error);
            throw error;
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
    static getChatbot(vocabSize = 1000, maxSeqLength = 20) {
        // Simple Q&A chatbot dataset
        const conversations = [
            { input: "hello", output: "Hello! How can I help you today?" },
            { input: "how are you", output: "I'm doing great, thanks for asking!" },
            { input: "what is your name", output: "I'm a chatbot trained to help you." },
            { input: "goodbye", output: "Goodbye! Have a great day!" },
            { input: "thank you", output: "You're welcome!" },
            { input: "help", output: "I'm here to help! Ask me anything." },
            { input: "what can you do", output: "I can chat with you and answer questions." },
            { input: "how old are you", output: "I'm as old as code that created me!" },
            { input: "where are you from", output: "I exist in digital world." },
            { input: "tell me a joke", output: "Why don't scientists trust atoms? Because they make up everything!" }
        ];
        
        return this.processChatData(conversations, vocabSize, maxSeqLength);
    }
    
    static getConversation(vocabSize = 1000, maxSeqLength = 20) {
        // Extended conversation pairs
        const conversations = [
            { input: "hi there", output: "Hello! Nice to meet you." },
            { input: "how's it going", output: "It's going well! How about you?" },
            { input: "what's up", output: "Not much, just here to chat!" },
            { input: "nice to meet you", output: "Nice to meet you too!" },
            { input: "how was your day", output: "Every day is a good day when I get to help people!" },
            { input: "what do you like", output: "I like helping people and having interesting conversations." },
            { input: "what's the weather", output: "I don't have access to weather data, but I hope it's nice where you are!" },
            { input: "are you real", output: "I'm as real as the conversations we have!" },
            { input: "can you learn", output: "I can learn from our conversations and improve over time." },
            { input: "what time is it", output: "I don't have access to a clock, but time flies when you're having fun!" }
        ];
        
        return this.processChatData(conversations, vocabSize, maxSeqLength);
    }
    
    static processChatData(conversations, vocabSize = 1000, maxSeqLength = 20) {
        // Build vocabulary
        const vocab = new Set();
        conversations.forEach(conv => {
            conv.input.split(' ').forEach(word => vocab.add(word.toLowerCase()));
            conv.output.split(' ').forEach(word => vocab.add(word.toLowerCase()));
        });
        
        const vocabArray = Array.from(vocab).slice(0, vocabSize);
        const wordToIndex = {};
        vocabArray.forEach((word, index) => {
            wordToIndex[word] = index;
        });
        
        // Convert conversations to numerical data
        const trainingData = [];
        conversations.forEach(conv => {
            const inputSeq = this.textToSequence(conv.input, wordToIndex, maxSeqLength);
            const outputSeq = this.textToSequence(conv.output, wordToIndex, maxSeqLength);
            
            trainingData.push({
                input: inputSeq.map(val => [val]),
                output: outputSeq.map(val => [val])
            });
        });
        
        // Split into train/test
        const splitIndex = Math.floor(trainingData.length * 0.8);
        const trainData = trainingData.slice(0, splitIndex);
        const testData = trainingData.slice(splitIndex);
        
        return { 
            trainingData: trainData, 
            testData: testData,
            vocab: vocabArray,
            wordToIndex: wordToIndex,
            indexToWord: vocabArray
        };
    }
    
    static textToSequence(text, wordToIndex, maxLength) {
        const words = text.toLowerCase().split(' ');
        const sequence = [];
        
        for (let i = 0; i < Math.min(words.length, maxLength); i++) {
            const word = words[i];
            sequence.push(wordToIndex[word] || 0); // 0 for unknown words
        }
        
        // Pad sequence if needed
        while (sequence.length < maxLength) {
            sequence.push(0); // 0 for padding
        }
        
        return sequence;
    }
    
static sequenceToText(sequence, indexToWord) {
    return sequence
        .filter(index => index > 0) // Remove padding
        .map(index => indexToWord[index] || '<unk>')
        .join(' ');
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
        document.getElementById('sendBtn').addEventListener('click', () => this.sendMessage());
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        // Slider event listeners
        const accuracyGoalSlider = document.getElementById('accuracyGoal');
        const patienceSlider = document.getElementById('patience');
        
        accuracyGoalSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('accuracyGoalValue').textContent = value.toFixed(1) + '%';
        });
        
        patienceSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('patienceValue').textContent = value + ' epochs';
        });
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
            const datasetType = document.getElementById('datasetType').value;
            
            // For chatbot models, use vocabulary and sequence length
            let inputSize, outputSize;
            
            if (datasetType === 'chatbot' || datasetType === 'conversation') {
                inputSize = parseInt(document.getElementById('maxSeqLength').value);
                outputSize = parseInt(document.getElementById('maxSeqLength').value);
                this.log(`🤖 Setting up chatbot network with sequence length: ${inputSize}`, 'info');
            } else {
                inputSize = parseInt(document.getElementById('inputSize').value);
                outputSize = parseInt(document.getElementById('outputSize').value);
            }
            
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
            
            // Load dataset first
            this.loadDataset();
            
            // Only setup test inputs for non-chatbot models
            if (datasetType !== 'chatbot' && datasetType !== 'conversation') {
                this.setupTestInputs();
            }
            
            // Enable training button
            document.getElementById('trainBtn').disabled = false;
            
        } catch (error) {
            this.log(`❌ Error initializing network: ${error.message}`, 'error');
        }
    }

    loadDataset() {
        const datasetType = document.getElementById('datasetType').value;
        
        switch (datasetType) {
            case 'chatbot':
                const vocabSize = parseInt(document.getElementById('vocabSize').value || 1000);
                const maxSeqLength = parseInt(document.getElementById('maxSeqLength').value || 20);
                this.dataset = Dataset.getChatbot(vocabSize, maxSeqLength);
                this.log('📚 Loaded Simple Chatbot dataset', 'info');
                break;
            case 'conversation':
                const convVocabSize = parseInt(document.getElementById('vocabSize').value || 1000);
                const convMaxSeqLength = parseInt(document.getElementById('maxSeqLength').value || 20);
                this.dataset = Dataset.getConversation(convVocabSize, convMaxSeqLength);
                this.log('📚 Loaded Conversation Pairs dataset', 'info');
                break;
            case 'custom':
                // Custom data will be loaded via file upload
                break;
            default:
                this.dataset = Dataset.getChatbot(1000, 20);
        }
        
        if (this.dataset && datasetType !== 'custom') {
            this.log(`📊 Dataset loaded: ${this.dataset.trainingData.length} training samples`, 'info');
            this.log(`📝 Vocabulary size: ${this.dataset.vocab.length} words`, 'info');
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
        const accuracyGoal = parseFloat(document.getElementById('accuracyGoal').value);
        const maxPatience = parseInt(document.getElementById('patience').value);
        let currentLearningRate = learningRate;
        
        // Add crash detection
        this.crashDetectionTimer = setTimeout(() => {
            if (this.isTraining) {
                this.log('💥 Training appears to have crashed or frozen', 'error');
                this.stopTraining();
            }
        }, 30000); // 30 seconds timeout
        
        try {
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
                
                // Auto-stop if target accuracy reached
                if (metrics.accuracy >= accuracyGoal) {
                    this.log(`🎯 Target accuracy reached: ${metrics.accuracy.toFixed(2)}%`, 'success');
                    break;
                }
                
                // Reset crash detection timer
                clearTimeout(this.crashDetectionTimer);
                this.crashDetectionTimer = setTimeout(() => {
                    if (this.isTraining) {
                        this.log('💥 Training appears to have crashed or frozen', 'error');
                        this.stopTraining();
                    }
                }, 30000);
                
                // Memory cleanup
                if (epoch % 10 === 0) {
                    this.cleanupMemory();
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            }
        } catch (error) {
            this.log(`💥 Training crashed: ${error.message}`, 'error');
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
        
        try {
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
            
            const accuracy = (correct / testData.length) * 100;
            const loss = totalLoss / testData.length;
            
            // Update UI in real-time
            document.getElementById('trainingLoss').textContent = loss.toFixed(4);
            document.getElementById('validationLoss').textContent = loss.toFixed(4);
            document.getElementById('accuracy').textContent = accuracy.toFixed(2) + '%';
            
            return { loss, accuracy };
        } catch (error) {
            this.log(`❌ Error calculating metrics: ${error.message}`, 'error');
            return { loss: 0, accuracy: 0 };
        }
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
        const accuracyGoal = parseFloat(document.getElementById('accuracyGoal').value);
        
        // Update all meters every 500ms for smoother updates
        this.progressInterval = setInterval(() => {
            if (this.isTraining && this.currentEpoch > 0) {
                const progress = (this.currentEpoch / totalEpochs) * 100;
                const metrics = this.calculateMetrics();
                
                // Update progress bar
                this.updateProgressBar(progress, `Epoch ${this.currentEpoch}/${totalEpochs} - Accuracy: ${metrics.accuracy.toFixed(1)}% (Goal: ${accuracyGoal.toFixed(1)}%)`);
                
                // Update all UI elements in real-time
                this.updateAllMeters(metrics, progress);
                
                // Log accuracy improvement every second
                if (metrics.accuracy > 0) {
                    console.log(`🎯 Current accuracy: ${metrics.accuracy.toFixed(2)}% (Target: ${accuracyGoal.toFixed(1)}%)`);
                }
            }
        }, 500); // Update every 500ms for smoother experience
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    updateAllMeters(metrics, progress) {
        // Update all training metrics in real-time
        document.getElementById('currentEpoch').textContent = this.currentEpoch;
        document.getElementById('trainingLoss').textContent = metrics.loss.toFixed(4);
        document.getElementById('validationLoss').textContent = metrics.loss.toFixed(4);
        document.getElementById('accuracy').textContent = metrics.accuracy.toFixed(2) + '%';
        
        // Update training time
        if (this.trainingStartTime) {
            const elapsed = Date.now() - this.trainingStartTime;
            const seconds = Math.floor(elapsed / 1000) % 60;
            const minutes = Math.floor(elapsed / 60000) % 60;
            const hours = Math.floor(elapsed / 3600000);
            
            document.getElementById('trainingTime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // Update progress bar with color coding
        this.updateProgressBar(progress, `Epoch ${this.currentEpoch} - Loss: ${metrics.loss.toFixed(4)} - Acc: ${metrics.accuracy.toFixed(1)}%`);
        
        // Add visual indicators for performance
        this.updatePerformanceIndicators(metrics);
    }

    updatePerformanceIndicators(metrics) {
        // Add color coding and indicators based on performance
        const accuracyElement = document.getElementById('accuracy');
        const lossElement = document.getElementById('trainingLoss');
        const validationLossElement = document.getElementById('validationLoss');
        
        // Remove existing performance classes
        [accuracyElement, lossElement, validationLossElement].forEach(el => {
            el.classList.remove('high-performance', 'medium-performance', 'low-performance');
        });
        
        // Color code accuracy
        if (metrics.accuracy >= 90) {
            accuracyElement.classList.add('high-performance');
        } else if (metrics.accuracy >= 70) {
            accuracyElement.classList.add('medium-performance');
        } else {
            accuracyElement.classList.add('low-performance');
        }
        
        // Color code loss
        if (metrics.loss <= 0.1) {
            lossElement.classList.add('high-performance');
            validationLossElement.classList.add('high-performance');
        } else if (metrics.loss <= 0.5) {
            lossElement.classList.add('medium-performance');
            validationLossElement.classList.add('medium-performance');
        } else {
            lossElement.classList.add('low-performance');
            validationLossElement.classList.add('low-performance');
        }
        
        // Add performance badges
        this.updatePerformanceBadges(metrics);
    }

    updatePerformanceBadges(metrics) {
        // Add visual badges for performance indicators
        let badgeText = '';
        let badgeClass = '';
        
        if (metrics.accuracy >= 95) {
            badgeText = '🏆 EXCELLENT';
            badgeClass = 'high-performance';
        } else if (metrics.accuracy >= 80) {
            badgeText = '⭐ GOOD';
            badgeClass = 'medium-performance';
        } else if (metrics.accuracy >= 60) {
            badgeText = '📈 LEARNING';
            badgeClass = 'medium-performance';
        } else {
            badgeText = '🔄 TRAINING';
            badgeClass = 'low-performance';
        }
        
        // Update or create badge
        let badge = document.getElementById('performanceBadge');
        if (!badge) {
            badge = document.createElement('div');
            badge.id = 'performanceBadge';
            badge.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                border-radius: 25px;
                font-weight: bold;
                font-size: 14px;
                z-index: 1000;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            `;
            document.body.appendChild(badge);
        }
        
        badge.textContent = badgeText;
        badge.className = badgeClass;
        
        // Color the badge
        if (badgeClass === 'high-performance') {
            badge.style.background = 'linear-gradient(135deg, #38a169 0%, #48bb78 100%)';
            badge.style.color = 'white';
        } else if (badgeClass === 'medium-performance') {
            badge.style.background = 'linear-gradient(135deg, #d69e2e 0%, #f6e05e 100%)';
            badge.style.color = 'white';
        } else {
            badge.style.background = 'linear-gradient(135deg, #e53e3e 0%, #fc8181 100%)';
            badge.style.color = 'white';
        }
    }

    updateProgressBar(percentage, text) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressFill.style.width = `${Math.min(percentage, 100)}%`;
        progressText.textContent = text;
        
        // Change color based on progress and performance
        const metrics = this.calculateMetrics();
        if (metrics.accuracy >= 90) {
            progressFill.style.background = 'linear-gradient(90deg, #38a169 0%, #48bb78 100%)';
        } else if (percentage >= 50) {
            progressFill.style.background = 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)';
        } else {
            progressFill.style.background = 'linear-gradient(90deg, #ed8936 0%, #f6ad55 100%)';
        }
        
        // Add pulse animation for active training
        if (this.isTraining) {
            progressFill.style.animation = 'pulse 2s infinite';
        } else {
            progressFill.style.animation = 'none';
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

    sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || !this.network || !this.dataset) {
            return;
        }
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Generate response
        const response = this.generateResponse(message);
        this.addMessage(response, 'bot');
        
        // Clear input
        input.value = '';
    }
    
    generateResponse(message) {
        try {
            // Convert message to sequence
            const sequence = Dataset.textToSequence(
                message, 
                this.dataset.wordToIndex, 
                parseInt(document.getElementById('maxSeqLength').value)
            );
            
            // Get model prediction
            const input = sequence.map(val => [val]);
            const output = this.network.forward(input);
            
            // Convert output back to text
            const outputSequence = output.map(val => Math.round(val[0]));
            const response = Dataset.sequenceToText(outputSequence, this.dataset.indexToWord);
            
            // If no meaningful response, use fallback
            if (!response || response.trim() === '') {
                return this.getFallbackResponse(message);
            }
            
            return response;
        } catch (error) {
            console.error('Error generating response:', error);
            return this.getFallbackResponse(message);
        }
    }
    
    getFallbackResponse(message) {
        const fallbacks = [
            "That's interesting! Tell me more.",
            "I see. What else would you like to discuss?",
            "Thanks for sharing that with me.",
            "I'm still learning, but I appreciate your message!",
            "That's a good point. How else can I help you?",
            "I'm processing what you said. Can you elaborate?"
        ];
        
        // Simple keyword-based responses
        const lowerMessage = message.toLowerCase();
        if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
            return "Hello there! How can I help you today?";
        }
        if (lowerMessage.includes('how are')) {
            return "I'm doing great, thanks for asking!";
        }
        if (lowerMessage.includes('goodbye') || lowerMessage.includes('bye')) {
            return "Goodbye! Have a great day!";
        }
        if (lowerMessage.includes('thank')) {
            return "You're welcome!";
        }
        
        // Random fallback
        return fallbacks[Math.floor(Math.random() * fallbacks.length)];
    }
    
    addMessage(text, sender) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const senderLabel = sender === 'user' ? 'You' : 'AI';
        messageDiv.innerHTML = `<strong>${senderLabel}:</strong> ${text}`;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new AITrainer();
});
