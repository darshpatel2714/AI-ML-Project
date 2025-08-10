
        let model = null;
        let trainingHistory = [];
        let scaleParams = null;

        // Update location slider display
        document.getElementById('location').addEventListener('input', function() {
            document.getElementById('locationValue').textContent = this.value;
        });

        // Generate synthetic training data
        function generateTrainingData(numSamples) {
            const data = [];
            
            for (let i = 0; i < numSamples; i++) {
                const bedrooms = Math.floor(Math.random() * 5) + 1;
                const bathrooms = Math.floor(Math.random() * 4) + 1;
                const sqft = Math.floor(Math.random() * 3000) + 800;
                const age = Math.floor(Math.random() * 50);
                const location = Math.floor(Math.random() * 10) + 1;
                
                // Price calculation with realistic factors
                let price = 50000; // Base price
                price += sqft * 1500; // ₹150 per sqft
                price += bedrooms * 250000; // ₹25k per bedroom
                price += bathrooms * 150000; // ₹15k per bathroom
                price -= age * 20000; // Depreciation
                price += location * 300000; // Location premium
                
                // Add some noise
                price += (Math.random() - 0.5) * 1000000;
                price = Math.max(price, 800000); // Minimum price
                
                data.push({
                    features: [bedrooms, bathrooms, sqft, age, location],
                    price: price
                });
            }
            
            return data;
        }

        // Normalize data
        function normalizeData(data) {
            const features = data.map(d => d.features);
            const prices = data.map(d => d.price);
            
            // Calculate min/max for features
            const featureMins = features[0].map(() => Infinity);
            const featureMaxs = features[0].map(() => -Infinity);
            
            features.forEach(feature => {
                feature.forEach((value, index) => {
                    featureMins[index] = Math.min(featureMins[index], value);
                    featureMaxs[index] = Math.max(featureMaxs[index], value);
                });
            });
            
            const priceMin = Math.min(...prices);
            const priceMax = Math.max(...prices);
            
            scaleParams = {
                featureMins,
                featureMaxs,
                priceMin,
                priceMax
            };
            
            // Normalize features and prices
            const normalizedFeatures = features.map(feature =>
                feature.map((value, index) =>
                    (value - featureMins[index]) / (featureMaxs[index] - featureMins[index])
                )
            );
            
            const normalizedPrices = prices.map(price =>
                (price - priceMin) / (priceMax - priceMin)
            );
            
            return {
                features: normalizedFeatures,
                prices: normalizedPrices
            };
        }

        // Start training process
        async function startTraining() {
            const trainBtn = document.getElementById('trainBtn');
            const trainText = document.getElementById('trainText');
            const trainLoader = document.getElementById('trainLoader');
            const progressFill = document.getElementById('progressFill');
            const trainingStatus = document.getElementById('trainingStatus');
            
            trainBtn.disabled = true;
            trainText.style.display = 'none';
            trainLoader.style.display = 'inline-block';
            trainingStatus.textContent = 'Preparing training data...';
            trainingHistory = [];
            
            try {
                // Generate training data
                const rawData = generateTrainingData(500);
                const normalizedData = normalizeData(rawData);
                
                trainingStatus.textContent = 'Building neural network...';
                progressFill.style.width = '10%';
                
                // Create a simple model
                model = tf.sequential();
                model.add(tf.layers.dense({
                    inputShape: [5],
                    units: 20,
                    activation: 'relu'
                }));
                model.add(tf.layers.dense({
                    units: 10,
                    activation: 'relu'
                }));
                model.add(tf.layers.dense({
                    units: 1
                }));
                
                model.compile({
                    optimizer: 'sgd',
                    loss: 'meanSquaredError'
                });
                
                trainingStatus.textContent = 'Training the AI model...';
                progressFill.style.width = '30%';
                
                // Prepare training data
                const trainX = tf.tensor2d(normalizedData.features);
                const trainY = tf.tensor2d(normalizedData.prices.map(p => [p]));
                
                // Train with fewer epochs for stability
                const epochs = 30;
                for (let epoch = 0; epoch < epochs; epoch++) {
                    const history = await model.fit(trainX, trainY, {
                        epochs: 1,
                        batchSize: 8,
                        shuffle: true,
                        verbose: 0
                    });
                    
                    const loss = history.history.loss[0];
                    trainingHistory.push({loss: loss, val_loss: loss});
                    
                    const progress = 30 + (epoch / epochs) * 70;
                    progressFill.style.width = progress + '%';
                    trainingStatus.textContent = 'Training epoch ' + (epoch + 1) + '/' + epochs + ' (Loss: ' + loss.toFixed(4) + ')';
                    
                    // Small delay to show progress
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
                
                trainX.dispose();
                trainY.dispose();
                
                progressFill.style.width = '100%';
                trainingStatus.textContent = 'Training complete! Ready to predict house prices.';
                
                const finalLoss = trainingHistory[trainingHistory.length - 1].loss;
                const accuracy = Math.max(0, Math.min(100, (1 - finalLoss) * 100)).toFixed(1);
                document.getElementById('modelAccuracy').textContent = accuracy + '%';
                
                document.getElementById('predictBtn').disabled = false;
                updateChart();
                
            } catch (error) {
                trainingStatus.textContent = 'Training failed. Please refresh and try again.';
                console.log('Error details:', error.message);
            } finally {
                trainBtn.disabled = false;
                trainText.style.display = 'inline';
                trainLoader.style.display = 'none';
            }
        }

        // Make price prediction
        async function makePrediction() {
            if (!model || !scaleParams) {
                alert('Please train the model first!');
                return;
            }
            
            try {
                const bedrooms = parseInt(document.getElementById('bedrooms').value);
                const bathrooms = parseInt(document.getElementById('bathrooms').value);
                const sqft = parseInt(document.getElementById('sqft').value);
                const age = parseInt(document.getElementById('age').value);
                const location = parseInt(document.getElementById('location').value);
                
                // Normalize input
                const normalizedInput = [
                    (bedrooms - scaleParams.featureMins[0]) / (scaleParams.featureMaxs[0] - scaleParams.featureMins[0]),
                    (bathrooms - scaleParams.featureMins[1]) / (scaleParams.featureMaxs[1] - scaleParams.featureMins[1]),
                    (sqft - scaleParams.featureMins[2]) / (scaleParams.featureMaxs[2] - scaleParams.featureMins[2]),
                    (age - scaleParams.featureMins[3]) / (scaleParams.featureMaxs[3] - scaleParams.featureMins[3]),
                    (location - scaleParams.featureMins[4]) / (scaleParams.featureMaxs[4] - scaleParams.featureMins[4])
                ];
                
                // Make prediction
                const inputTensor = tf.tensor2d([normalizedInput]);
                const prediction = model.predict(inputTensor);
                
                // Get prediction result
                const predArray = await prediction.data();
                const normalizedPrice = predArray[0];
                const actualPrice = normalizedPrice * (scaleParams.priceMax - scaleParams.priceMin) + scaleParams.priceMin;
                
                document.getElementById('priceResult').textContent = '₹' + Math.round(actualPrice).toLocaleString();
                
                // Calculate confidence
                const confidence = Math.min(95, Math.max(70, 90 - Math.abs(normalizedPrice - 0.5) * 40));
                document.getElementById('confidenceLevel').textContent = 'Model Confidence: ' + confidence.toFixed(1) + '%';
                
                // Clean up tensors
                inputTensor.dispose();
                prediction.dispose();
                
            } catch (error) {
                console.log('Prediction error:', error);
                document.getElementById('priceResult').textContent = 'Error in prediction';
            }
        }

        // Update training chart
        function updateChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            // Destroy existing chart if it exists
            if (window.myChart && typeof window.myChart.destroy === 'function') {
                window.myChart.destroy();
            }
            
            const epochs = trainingHistory.map((_, i) => i + 1);
            const losses = trainingHistory.map(h => h.loss);
            
            window.myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [{
                        label: 'Training Loss',
                        data: losses,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Model Training Progress',
                            color: 'white'
                        },
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Training Epochs',
                                color: 'white'
                            },
                            ticks: {
                                color: 'white'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss Value',
                                color: 'white'
                            },
                            ticks: {
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('House Price Predictor AI Model Ready!');
        });
  