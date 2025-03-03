class LucidRecallClient {
    constructor() {
        // Adjust these URLs/ports to match your HPC & Tensor servers
        this.tensorServer = new WebSocket('ws://localhost:5001');
        this.hpcServer = new WebSocket('ws://localhost:5004');
        this.lmStudio = null;
        this.memoryCount = 0;
        this.selectedMemories = new Set();
        this.memoryContext = [];
        
        // Local cache (timestamp → embeddings)
        this.memoryCache = new Map();
        
        // Bind methods
        this.processEmbedding = this.processEmbedding.bind(this);
        this.showMemoryResults = this.showMemoryResults.bind(this);
        this.updateMemoryStats = this.updateMemoryStats.bind(this);
        this.toggleMemorySelection = this.toggleMemorySelection.bind(this);
        
        this.setupEventListeners();
        this.setupWebSockets();
        this.initializeNeuralBackground();
    }

    // --------------------------
    // WebSocket Setup
    // --------------------------
    setupWebSockets() {
        // --------------------------
        // Tensor Server
        // --------------------------
        this.tensorServer.onopen = () => {
            console.log("Tensor Server Connected");
            this.updateStatus('Connected');
            this.addMessage('system', 'Memory system connected');
        };

        this.tensorServer.onmessage = async (event) => {
            console.log("Tensor Server received:", event.data);
            const data = JSON.parse(event.data);
            console.log("Parsed tensor data:", data);
            
            if (data.type === 'embeddings') {
                console.log("Processing embeddings:", data.embeddings);
                await this.processEmbedding(data.embeddings);
            } else if (data.type === 'search_results') {
                console.log("Search results:", data.results);
                this.showMemoryResults(data.results);
            } else if (data.type === 'stats') {
                console.log("Memory stats:", data);
                this.updateMemoryStats(data);
            }
        };

        this.tensorServer.onclose = () => {
            console.log("Tensor Server Disconnected");
            this.updateStatus('Disconnected');
            this.addMessage('system', 'Memory system disconnected');
        };

        this.tensorServer.onerror = (error) => {
            console.error('Tensor Server error:', error);
            this.addMessage('system', 'Memory system connection error');
        };

        // --------------------------
        // HPC Server
        // --------------------------
        this.hpcServer.onopen = () => {
            console.log("HPC Server Connected");
            this.addMessage('system', 'HPC system connected');
        };

        this.hpcServer.onmessage = (event) => {
            console.log("HPC Server received:", event.data);
            const data = JSON.parse(event.data);
            console.log("Parsed HPC data:", data);

            // HPC server responds with { type: "processed" } on success
            if (data.type === 'processed') {
                // Increase local memory count or do whatever you want with HPC results
                this.memoryCount++;
                document.getElementById('memory-count').textContent = this.memoryCount;
            } else if (data.type === 'error') {
                // HPC server might respond with an error
                this.addMessage('system', `HPC error: ${data.error}`);
            }
        };

        this.hpcServer.onclose = () => {
            console.log("HPC Server Disconnected");
            this.addMessage('system', 'HPC system disconnected');
        };

        this.hpcServer.onerror = (error) => {
            console.error('HPC Server error:', error);
            this.addMessage('system', 'HPC system connection error');
        };
    }

    // --------------------------
    // Embedding => HPC
    // --------------------------
    async processEmbedding(embeddings) {
        try {
            console.log("Processing embedding:", embeddings);
            // Send to HPC server
            if (this.hpcServer.readyState === WebSocket.OPEN) {
                // HPC expects { type: "process", embeddings: [...] }
                const hpcRequest = {
                    type: 'process',
                    embeddings: embeddings
                };
                console.log("Sending to HPC server:", hpcRequest);
                this.hpcServer.send(JSON.stringify(hpcRequest));
            } else {
                console.warn("HPC server not connected");
            }
            
            // Also cache locally with a timestamp
            const timestamp = Date.now();
            console.log("Caching embedding at timestamp:", timestamp);
            this.memoryCache.set(timestamp, embeddings);
            
            // Indicate some status in the UI
            this.updateStatus('Processing');
        } catch (error) {
            console.error('Error processing embedding:', error);
            this.addMessage('system', 'Error processing memory');
        }
    }

    // --------------------------
    // User Input => Store => Search => LM
    // --------------------------
    async processInput() {
        const input = document.getElementById('input');
        const text = input.value.trim();
        if (!text) return;

        // Show user message
        this.addMessage('user', text);
        input.value = '';

        // If you want context from selected memories
        let context = '';
        if (this.memoryContext.length > 0) {
            context =
                'Related Memories:\n' + 
                this.memoryContext.map(m => m.text).join('\n\n') +
                '\n\n';
        }

        // Store in memory (on Tensor server)
        this.tensorServer.send(JSON.stringify({
            type: 'embed',
            text: text
        }));

        // Search memory
        this.tensorServer.send(JSON.stringify({
            type: 'search',
            text: text,
            limit: 5
        }));

        // (Optional) Send to LM Studio
        if (this.lmStudio) {
            try {
                const response = await fetch('http://127.0.0.1:1234/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{
                            role: "user",
                            content: context + text
                        }],
                        model: "qwen2.5-7b-instruct",
                        temperature: 0.7,
                        max_tokens: 500,
                        stream: false
                    })
                });

                const result = await response.json();
                if (result.choices && result.choices[0].message) {
                    const message = result.choices[0].message.content;
                    this.addMessage('assistant', message);

                    // Also store the assistant's response
                    this.tensorServer.send(JSON.stringify({
                        type: 'embed',
                        text: message
                    }));
                }
            } catch (error) {
                console.error('LM Studio error:', error);
                this.addMessage('system', 'Error getting LM Studio response');
            }
        }

        // Clear memory selection
        this.selectedMemories.clear();
        this.memoryContext = [];
        document.querySelectorAll('.memory-selection.selected').forEach(el => {
            el.classList.remove('selected');
        });
    }

    // --------------------------
    // Display Memory Search Results
    // --------------------------
    showMemoryResults(results) {
        if (!results || !results.length) return;

        const memoryMessage = document.createElement('div');
        memoryMessage.className = 'message memory';

        results.forEach(result => {
            const memoryDiv = document.createElement('div');
            memoryDiv.className = 'memory-selection';
            
            const similarity = (result.similarity * 100).toFixed(1);
            memoryDiv.innerHTML = `
                <span class="memory-match">${similarity}% match</span>
                <span class="memory-text">${result.text}</span>
                <span class="memory-significance">Significance: ${result.significance.toFixed(3)}</span>
            `;

            memoryDiv.addEventListener('click', () => {
                this.toggleMemorySelection(result, memoryDiv);
            });

            memoryMessage.appendChild(memoryDiv);
        });

        const container = document.getElementById('chat-container');
        container.appendChild(memoryMessage);
        container.scrollTop = container.scrollHeight;
    }

    // --------------------------
    // Handle Memory Selection
    // --------------------------
    toggleMemorySelection(memory, element) {
        const memoryId = memory.id;
        
        if (this.selectedMemories.has(memoryId)) {
            this.selectedMemories.delete(memoryId);
            element.classList.remove('selected');
            this.memoryContext = this.memoryContext.filter(m => m.id !== memoryId);
        } else {
            this.selectedMemories.add(memoryId);
            element.classList.add('selected');
            this.memoryContext.push(memory);
        }

        this.updateStatus(`${this.selectedMemories.size} memories selected`);
    }

    // --------------------------
    // Memory Stats
    // --------------------------
    updateMemoryStats(stats) {
        console.log("Updating memory stats:", stats);
        document.getElementById('memory-metric').textContent = 
            `${(stats.gpu_memory || 0).toFixed(2)} GB`;
        if (stats.memory_count !== undefined) {
            this.memoryCount = stats.memory_count;
            document.getElementById('memory-count').textContent = this.memoryCount;
        }
    }

    // --------------------------
    // UI Listeners
    // --------------------------
    setupEventListeners() {
        document.getElementById('send').addEventListener('click', () => this.processInput());
        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.processInput();
            }
        });
        document.getElementById('connect-lm').addEventListener('click', () => this.connectLMStudio());
    }

    // --------------------------
    // LM Studio Connection (Optional)
    // --------------------------
    async connectLMStudio() {
        if (this.lmStudio) {
            console.log("Disconnecting from LM Studio");
            this.lmStudio = null;
            document.getElementById('lm-status').classList.remove('connected');
            document.getElementById('connect-lm').classList.remove('active');
            return;
        }

        try {
            console.log("Connecting to LM Studio");
            const response = await fetch('http://127.0.0.1:1234/v1/models');
            const models = await response.json();
            console.log('Available LM Studio models:', models);

            document.getElementById('lm-status').classList.add('connected');
            document.getElementById('connect-lm').classList.add('active');
            this.addMessage('system', 'LM Studio connected');
            this.lmStudio = true;
        } catch (error) {
            console.error('LM Studio connection error:', error);
            this.addMessage('system', 'Failed to connect to LM Studio');
        }
    }

    // --------------------------
    // Add Message to Chat
    // --------------------------
    addMessage(type, content) {
        console.log(`Adding message of type ${type}:`, content);
        const container = document.getElementById('chat-container');
        const message = document.createElement('div');
        message.className = `message ${type}`;
        
        const md = window.markdownit({
            html: true,
            linkify: true,
            typographer: true
        });
        message.innerHTML = md.render(content);
        
        container.appendChild(message);
        container.scrollTop = container.scrollHeight;
    }

    // --------------------------
    // Update Status
    // --------------------------
    updateStatus(status) {
        console.log("Updating status:", status);
        document.getElementById('status-metric').textContent = status;
    }

    // --------------------------
    // Animated Background
    // --------------------------
    initializeNeuralBackground() {
        const canvas = document.getElementById('neural-bg');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const nodes = [];
        const nodeCount = 50;

        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2
            });
        }

        function animate() {
            ctx.fillStyle = 'rgba(10, 10, 31, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            nodes.forEach(node => {
                node.x += node.vx;
                node.y += node.vy;

                if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
                if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

                ctx.beginPath();
                ctx.arc(node.x, node.y, 2, 0, Math.PI * 2);
                ctx.fillStyle = '#00ffff';
                ctx.fill();
            });

            nodes.forEach((node1, i) => {
                nodes.slice(i + 1).forEach(node2 => {
                    const dx = node2.x - node1.x;
                    const dy = node2.y - node1.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(node1.x, node1.y);
                        ctx.lineTo(node2.x, node2.y);
                        ctx.strokeStyle = `rgba(0, 255, 255, ${1 - distance / 100})`;
                        ctx.stroke();
                    }
                });
            });

            requestAnimationFrame(animate);
        }

        animate();
    }
}

// Initialize
window.addEventListener('load', () => new LucidRecallClient());
