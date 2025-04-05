# .gitignore

```
node_modules
dist
.DS_Store
server/public
vite.config.ts.*
*.tar.gz
```

# .replit

```
modules = ["nodejs-20", "web", "postgresql-16"]
run = "npm run dev"
hidden = [".config", ".git", "generated-icon.png", "node_modules", "dist"]

[nix]
channel = "stable-24_05"

[deployment]
deploymentTarget = "autoscale"
build = ["npm", "run", "build"]
run = ["npm", "run", "start"]

[[ports]]
localPort = 5000
externalPort = 80

[workflows]
runButton = "Project"

[[workflows.workflow]]
name = "Project"
mode = "parallel"
author = "agent"

[[workflows.workflow.tasks]]
task = "workflow.run"
args = "Start application"

[[workflows.workflow]]
name = "Start application"
author = "agent"

[workflows.workflow.metadata]
agentRequireRestartOnSave = false

[[workflows.workflow.tasks]]
task = "packager.installForAll"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "npm run dev"
waitForPort = 5000

```

# attached_assets\index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShadowWatch - Dark Web Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
        body {
            font-family: 'JetBrains Mono', monospace;
            background-color: #0a0a0a;
        }
        .signal-strength {
            height: 20px;
            width: 100px;
            position: relative;
        }
        .signal-bar {
            position: absolute;
            bottom: 0;
            width: 15px;
            background-color: #00ff00;
            opacity: 0.3;
        }
        .signal-bar.filled {
            opacity: 1;
        }
        .blink {
            animation: blink-animation 1.5s steps(2, start) infinite;
        }
        @keyframes blink-animation {
            to {
                visibility: hidden;
            }
        }
        .pulsing-border {
            animation: pulse-border 2s infinite;
        }
        @keyframes pulse-border {
            0% {
                box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(0, 255, 0, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(0, 255, 0, 0);
            }
        }
        .terminal-scroll {
            scrollbar-width: thin;
            scrollbar-color: #00ff00 #111111;
        }
        .terminal-scroll::-webkit-scrollbar {
            width: 8px;
        }
        .terminal-scroll::-webkit-scrollbar-track {
            background: #111111;
        }
        .terminal-scroll::-webkit-scrollbar-thumb {
            background-color: #00ff00;
            border-radius: 6px;
            border: 1px solid #00ff00;
        }
        .code-input {
            caret-color: #00ff00;
        }
        .gradient-text {
            background: linear-gradient(90deg, #00ff00, #00aa00);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .scan-animation {
            animation: scan 2s linear infinite;
        }
        @keyframes scan {
            0% {
                background-position: 0 0;
            }
            100% {
                background-position: 0 30px;
            }
        }
    </style>
</head>
<body class="text-gray-200">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <header class="flex justify-between items-center border-b border-green-600 pb-4 mb-8">
            <div class="flex items-center">
                <div class="w-4 h-12 bg-green-500 mr-3"></div>
                <div>
                    <h1 class="text-2xl font-bold gradient-text">SHADOWWATCH</h1>
                    <p class="text-xs text-gray-500">DARK WEB INTELLIGENCE PLATFORM</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <div class="flex items-center">
                    <div class="signal-strength mr-2">
                        <div class="signal-bar" style="left: 0; height: 5px;"></div>
                        <div class="signal-bar" style="left: 20px; height: 10px;"></div>
                        <div class="signal-bar filled" style="left: 40px; height: 15px;"></div>
                        <div class="signal-bar filled" style="left: 60px; height: 20px;"></div>
                        <div class="signal-bar filled" style="left: 80px; height: 20px;"></div>
                    </div>
                    <span class="text-green-500 text-xs">TOR RELAY: ACTIVE</span>
                </div>
                <span class="text-gray-500">|</span>
                <div class="text-xs">
                    <span class="text-yellow-400">ENCRYPTION: </span>
                    <span class="text-green-400">AES-256</span>
                </div>
                <span class="text-gray-500">|</span>
                <div class="text-xs">
                    <span class="text-yellow-400">SESSION: </span>
                    <span class="text-green-400">SECURE</span>
                </div>
                <button class="bg-red-900 hover:bg-red-800 text-xs px-3 py-1 rounded-md" id="killswitch">
                    <i class="fas fa-power-off mr-1"></i> KILLSWITCH
                </button>
            </div>
        </header>

        <!-- Main Dashboard -->
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <!-- Left sidebar -->
            <div class="lg:col-span-1 space-y-6">
                <!-- User Profile -->
                <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <div class="flex items-center mb-4">
                        <div class="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center border border-green-500">
                            <i class="fas fa-user-secret text-green-500"></i>
                        </div>
                        <div class="ml-3">
                            <h3 class="font-bold text-green-400">AGENT-74X</h3>
                            <p class="text-xs text-gray-500">SENIOR INTELLIGENCE ANALYST</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="bg-gray-800 rounded p-2">
                            <div class="text-gray-500">CLEARANCE</div>
                            <div class="text-green-400">LEVEL 5</div>
                        </div>
                        <div class="bg-gray-800 rounded p-2">
                            <div class="text-gray-500">STATUS</div>
                            <div class="text-green-400">ACTIVE</div>
                        </div>
                        <div class="bg-gray-800 rounded p-2">
                            <div class="text-gray-500">TEAM</div>
                            <div class="text-green-400">PHANTOM</div>
                        </div>
                        <div class="bg-gray-800 rounded p-2">
                            <div class="text-gray-500">SINCE</div>
                            <div class="text-green-400">2018</div>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 class="font-bold text-green-500 mb-3 flex items-center">
                        <i class="fas fa-bolt mr-2"></i> QUICK ACTIONS
                    </h3>
                    <div class="space-y-2">
                        <button id="deep-scan" class="w-full bg-gray-800 hover:bg-green-900 text-green-400 text-left text-sm py-2 px-3 rounded flex items-center">
                            <i class="fas fa-search mr-2"></i> DEEP SCAN
                        </button>
                        <button id="credential-trace" class="w-full bg-gray-800 hover:bg-green-900 text-green-400 text-left text-sm py-2 px-3 rounded flex items-center">
                            <i class="fas fa-fingerprint mr-2"></i> CREDENTIAL TRACE
                        </button>
                        <button id="chatter-analysis" class="w-full bg-gray-800 hover:bg-green-900 text-green-400 text-left text-sm py-2 px-3 rounded flex items-center">
                            <i class="fas fa-comment-dots mr-2"></i> CHATTER ANALYSIS
                        </button>
                        <button id="identity-alert" class="w-full bg-gray-800 hover:bg-green-900 text-green-400 text-left text-sm py-2 px-3 rounded flex items-center">
                            <i class="fas fa-mask mr-2"></i> IDENTITY ALERT
                        </button>
                    </div>
                    <div id="scan-progress" class="mt-3 hidden">
                        <div class="flex justify-between text-xs text-gray-400 mb-1">
                            <span>SCAN PROGRESS</span>
                            <span id="progress-percent">0%</span>
                        </div>
                        <div class="w-full bg-gray-800 rounded-full h-1.5">
                            <div id="progress-bar" class="bg-green-500 h-1.5 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                    <div id="scan-result" class="mt-3 hidden">
                        <div class="text-xs p-2 rounded border border-green-900 bg-gray-800 text-green-400">
                            <i class="fas fa-check-circle mr-1"></i>
                            <span>Scan completed successfully</span>
                        </div>
                    </div>
                </div>

                <!-- Threat Level -->
                <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <div class="flex justify-between items-center mb-3">
                        <h3 class="font-bold text-green-500 flex items-center">
                            <i class="fas fa-shield-alt mr-2"></i> THREAT LEVEL
                        </h3>
                        <span class="text-xs bg-red-900 text-red-300 px-2 py-1 rounded-full">ELEVATED</span>
                    </div>
                    <div class="h-2 w-full bg-gray-800 rounded-full mb-2">
                        <div class="h-2 bg-gradient-to-r from-green-500 to-yellow-500 rounded-full" style="width: 70%"></div>
                    </div>
                    <div class="flex justify-between text-xs text-gray-500">
                        <span>LOW</span>
                        <span>MODERATE</span>
                        <span>HIGH</span>
                        <span>CRITICAL</span>
                    </div>
                    <div class="mt-4 text-xs text-gray-400">
                        <div class="flex items-center mb-2">
                            <div class="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                            <span>+12% DARKNET ACTIVITY</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                            <span>3 NEW DATA BREACHES</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="lg:col-span-3 space-y-6">
                <!-- Active Threats -->
                <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
                    <div class="border-b border-gray-800 p-4 bg-gray-950 flex justify-between items-center">
                        <h2 class="font-bold text-green-400 flex items-center">
                            <i class="fas fa-radiation mr-2"></i> ACTIVE THREATS
                        </h2>
                        <div class="text-xs flex items-center">
                            <div class="w-2 h-2 rounded-full bg-green-500 mr-1 pulsing-border"></div>
                            <span class="text-green-400">REALTIME MONITORING</span>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-3">
                        <div class="border-b md:border-b-0 md:border-r border-gray-800 p-4 hover:bg-gray-950 cursor-pointer transition">
                            <div class="flex items-center mb-2">
                                <div class="w-9 h-9 bg-red-900 rounded-full flex items-center justify-center mr-3">
                                    <i class="fas fa-key text-red-400"></i>
                                </div>
                                <div>
                                    <h3 class="font-bold">CREDENTIAL LEAK</h3>
                                    <p class="text-xs text-gray-500">12 MINUTES AGO</p>
                                </div>
                            </div>
                            <p class="text-xs text-gray-400 mb-1">FINANCIAL INSTITUTION DATABASE</p>
                            <div class="flex justify-between text-xs mb-2">
                                <span class="text-red-400">26,752 RECORDS</span>
                                <span class="text-yellow-400">REDACTED HOST</span>
                            </div>
                            <div class="w-full bg-gray-800 rounded-full h-1.5">
                                <div class="bg-red-500 h-1.5 rounded-full" style="width: 85%"></div>
                            </div>
                        </div>
                        <div class="border-b md:border-b-0 md:border-r border-gray-800 p-4 hover:bg-gray-950 cursor-pointer transition">
                            <div class="flex items-center mb-2">
                                <div class="w-9 h-9 bg-yellow-900 rounded-full flex items-center justify-center mr-3">
                                    <i class="fas fa-comments text-yellow-400"></i>
                                </div>
                                <div>
                                    <h3 class="font-bold">CHATTER SPIKE</h3>
                                    <p class="text-xs text-gray-500">34 MINUTES AGO</p>
                                </div>
                            </div>
                            <p class="text-xs text-gray-400 mb-1">MENTION OF MAJOR RETAILER</p>
                            <div class="flex justify-between text-xs mb-2">
                                <span class="text-yellow-400">358 MENTIONS</span>
                                <span class="text-blue-400">72% INCREASE</span>
                            </div>
                            <div class="w-full bg-gray-800 rounded-full h-1.5">
                                <div class="bg-yellow-500 h-1.5 rounded-full" style="width: 65%"></div>
                            </div>
                        </div>
                        <div class="p-4 hover:bg-gray-950 cursor-pointer transition">
                            <div class="flex items-center mb-2">
                                <div class="w-9 h-9 bg-purple-900 rounded-full flex items-center justify-center mr-3">
                                    <i class="fas fa-server text-purple-400"></i>
                                </div>
                                <div>
                                    <h3 class="font-bold">SERVER BREACH</h3>
                                    <p class="text-xs text-gray-500">1 HOUR AGO</p>
                                </div>
                            </div>
                            <p class="text-xs text-gray-400 mb-1">GOVERNMENT CONTRACTOR</p>
                            <div class="flex justify-between text-xs mb-2">
                                <span class="text-purple-400">SENSITIVE DATA</span>
                                <span class="text-red-400">CONFIRMED</span>
                            </div>
                            <div class="w-full bg-gray-800 rounded-full h-1.5">
                                <div class="bg-purple-500 h-1.5 rounded-full" style="width: 45%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Terminal & Alerts -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div class="lg:col-span-2">
                        <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden h-full">
                            <div class="border-b border-gray-800 p-3 bg-gray-950 flex justify-between items-center">
                                <h2 class="font-bold text-green-400 flex items-center">
                                    <i class="fas fa-terminal mr-2"></i> MONITOR TERMINAL
                                </h2>
                                <div class="flex space-x-2">
                                    <button class="text-xs bg-gray-800 hover:bg-green-900 text-green-400 px-2 py-1 rounded" id="clear-terminal">
                                        <i class="fas fa-trash mr-1"></i> CLEAR
                                    </button>
                                    <button class="text-xs bg-gray-800 hover:bg-green-900 text-green-400 px-2 py-1 rounded" id="refresh-terminal">
                                        <i class="fas fa-redo mr-1"></i> REFRESH
                                    </button>
                                </div>
                            </div>
                            <div id="terminal-content" class="h-64 p-3 overflow-y-auto terminal-scroll bg-black text-green-400 text-xs font-mono">
                                <div class="mb-1"><span class="text-yellow-400">root@shadowwatch:~#</span> ./scan --deep --crawl -t credential_leak</div>
                                <div class="mb-1">[+] Initiating deep scan protocol...</div>
                                <div class="mb-1">[+] Connecting to TOR network... <span class="text-green-400">SUCCESS</span></div>
                                <div class="mb-1">[+] Establishing secure channels... <span class="text-green-400">SUCCESS</span></div>
                                <div class="mb-1">[+] Crawling 12 marketplaces, 8 forums, 15 channels</div>
                                <div class="mb-1">[!] Potential leak detected: financial_institution_db.tar.gz</div>
                                <div class="mb-1">[!] Analyzing contents... 
                                    <span class="inline-block loader animate-spin h-3 w-3 border-2 border-green-500 border-t-transparent rounded-full"></span>
                                </div>
                                <div class="mb-1">[!] CREDENTIAL ALERT: 26,752 records confirmed valid</div>
                                <div class="mb-1">[+] Initiating analysis of chatter patterns...</div>
                                <div class="mb-1">[!] Increased mentions of "major_retailer" detected (+72%)</div>
                                <div class="mb-1">[+] Scanning pastebin clones for matches...</div>
                                <div class="mb-1">[!] MATCH: server_dump.txt contains 438 credentials</div>
                                <div class="mb-1">root@shadowwatch:~#<span class="code-input"> _</span></div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden h-full">
                            <div class="border-b border-gray-800 p-3 bg-gray-950 flex justify-between items-center">
                                <h2 class="font-bold text-green-400 flex items-center">
                                    <i class="fas fa-bell mr-2"></i> ALERT FEED
                                </h2>
                                <span id="alert-count" class="text-xs bg-red-900 text-red-300 px-2 py-1 rounded-full">3 NEW</span>
                            </div>
                            <div id="alert-feed" class="h-64 overflow-y-auto terminal-scroll">
                                <div class="p-3 border-b border-gray-800 hover:bg-gray-950 cursor-pointer transition">
                                    <div class="flex justify-between items-start mb-1">
                                        <span class="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded-full">CRITICAL</span>
                                        <span class="text-xs text-gray-500">12:23:45</span>
                                    </div>
                                    <p class="text-sm">Large credential dump detected on DarkNet market</p>
                                    <p class="text-xs text-gray-400 mt-1 flex items-center">
                                        <i class="fas fa-database mr-1"></i> Financial records (26K+) potentially compromised
                                    </p>
                                </div>
                                <div class="p-3 border-b border-gray-800 hover:bg-gray-950 cursor-pointer transition">
                                    <div class="flex justify-between items-start mb-1">
                                        <span class="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded-full">WARNING</span>
                                        <span class="text-xs text-gray-500">11:58:12</span>
                                    </div>
                                    <p class="text-sm">Chatter spike detected mentioning major retailer</p>
                                    <p class="text-xs text-gray-400 mt-1 flex items-center">
                                        <i class="fas fa-comment-alt mr-1"></i> 358 mentions in last 2 hours (+72%)
                                    </p>
                                </div>
                                <div class="p-3 border-b border-gray-800 hover:bg-gray-950 cursor-pointer transition">
                                    <div class="flex justify-between items-start mb-1">
                                        <span class="text-xs bg-purple-900 text-purple-300 px-2 py-0.5 rounded-full">SUSPICIOUS</span>
                                        <span class="text-xs text-gray-500">10:35:27</span>
                                    </div>
                                    <p class="text-sm">Server breach reported on underground forum</p>
                                    <p class="text-xs text-gray-400 mt-1 flex items-center">
                                        <i class="fas fa-shield-alt mr-1"></i> Government contractor data potentially exposed
                                    </p>
                                </div>
                                <div class="p-3 border-b border-gray-800 hover:bg-gray-950 cursor-pointer transition">
                                    <div class="flex justify-between items-start mb-1">
                                        <span class="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded-full">INFO</span>
                                        <span class="text-xs text-gray-500">09:42:18</span>
                                    </div>
                                    <p class="text-sm">New ransomware variant detected in wild</p>
                                    <p class="text-xs text-gray-400 mt-1 flex items-center">
                                        <i class="fas fa-bug mr-1"></i> Targeting healthcare sector, requires patch KB5021234
                                    </p>
                                </div>
                                <div class="p-3 hover:bg-gray-950 cursor-pointer transition">
                                    <div class="flex justify-between items-start mb-1">
                                        <span class="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded-full">WARNING</span>
                                        <span class="text-xs text-gray-500">08:15:03</span>
                                    </div>
                                    <p class="text-sm">Unusual login attempts from ASN 14061</p>
                                    <p class="text-xs text-gray-400 mt-1 flex items-center">
                                        <i class="fas fa-globe mr-1"></i> 147 attempts from new IP blocks
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Data Analysis -->
                <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
                    <div class="border-b border-gray-800 p-4 bg-gray-950 flex justify-between items-center">
                        <h2 class="font-bold text-green-400 flex items-center">
                            <i class="fas fa-chart-line mr-2"></i> DARK WEB ACTIVITY ANALYSIS
                        </h2>
                        <div class="flex space-x-3">
                            <div class="text-xs">
                                <span class="text-yellow-400">LAST 24H: </span>
                                <span id="alert-count-main" class="text-green-400">1,247 ALERTS</span>
                            </div>
                            <div class="text-xs">
                                <span class="text-yellow-400">TOP THREAT: </span>
                                <span class="text-red-400">CREDENTIAL LEAKS</span>
                            </div>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
                        <div class="bg-gray-950 p-3 rounded border border-gray-800">
                            <div class="flex justify-between items-center mb-2">
                                <h3 class="text-sm font-bold text-green-400">CREDENTIAL EXPOSURE</h3>
                                <span class="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded-full">+24%</span>
                            </div>
                            <div class="h-40">
                                <canvas id="credentialChart"></canvas>
                            </div>
                            <div class="text-xs text-gray-400 mt-2 flex justify-between">
                                <span>12H AGO</span>
                                <span>6H AGO</span>
                                <span>CURRENT</span>
                            </div>
                        </div>
                        <div class="bg-gray-950 p-3 rounded border border-gray-800">
                            <div class="flex justify-between items-center mb-2">
                                <h3 class="text-sm font-bold text-green-400">FORUM ACTIVITY</h3>
                                <span class="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded-full">+18%</span>
                            </div>
                            <div class="h-40">
                                <canvas id="forumChart"></canvas>
                            </div>
                            <div class="text-xs text-gray-400 mt-2 flex justify-between">
                                <span>12H AGO</span>
                                <span>6H AGO</span>
                                <span>CURRENT</span>
                            </div>
                        </div>
                        <div class="bg-gray-950 p-3 rounded border border-gray-800">
                            <div class="flex justify-between items-center mb-2">
                                <h3 class="text-sm font-bold text-green-400">THREAT TYPES</h3>
                                <div class="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded-full">ANALYSIS</div>
                            </div>
                            <div class="h-40">
                                <canvas id="threatChart"></canvas>
                            </div>
                            <div class="text-xs text-gray-400 mt-2 flex justify-center space-x-4">
                                <span><span class="inline-block w-2 h-2 bg-red-500 rounded-full mr-1"></span>LEAKS</span>
                                <span><span class="inline-block w-2 h-2 bg-yellow-500 rounded-full mr-1"></span>CHATTER</span>
                                <span><span class="inline-block w-2 h-2 bg-purple-500 rounded-full mr-1"></span>BREACHES</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-8 border-t border-gray-800 pt-4 text-xs text-gray-600 flex justify-between items-center">
            <div>
                <span>SHADOWWATCH v3.7.4</span>
                <span class="mx-2">|</span>
                <span>SECURE SIGNAL: ENCRYPTED</span>
                <span class="mx-2">|</span>
                <span>LAST SYNCHRONIZATION: <span class="text-green-400">2m 34s ago</span></span>
            </div>
            <div class="flex items-center">
                <div class="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                <span>OPERATIONAL</span>
                <span class="mx-2">|</span>
                <span id="status-message" class="blink text-yellow-400">AWAITING COMMANDS</span>
            </div>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Initialize charts
        document.addEventListener('DOMContentLoaded', function() {
            // Credential Exposure Chart
            const ctx1 = document.getElementById('credentialChart').getContext('2d');
            new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', 'Now'],
                    datasets: [{
                        label: 'Credentials',
                        data: [65, 59, 80, 81, 92, 105, 132],
                        borderColor: '#ff5555',
                        backgroundColor: 'rgba(255, 85, 85, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: false,
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            display: false,
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });

            // Forum Activity Chart
            const ctx2 = document.getElementById('forumChart').getContext('2d');
            new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', 'Now'],
                    datasets: [{
                        label: 'Posts',
                        data: [30, 42, 55, 72, 63, 88, 102],
                        backgroundColor: 'rgba(255, 255, 85, 0.7)',
                        borderColor: 'rgba(255, 255, 85, 1)',
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: false,
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            display: false,
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });

            // Threat Types Chart
            const ctx3 = document.getElementById('threatChart').getContext('2d');
            new Chart(ctx3, {
                type: 'doughnut',
                data: {
                    labels: ['Credential Leaks', 'Chatter Spikes', 'Server Breaches'],
                    datasets: [{
                        data: [45, 35, 20],
                        backgroundColor: [
                            'rgba(255, 85, 85, 0.7)',
                            'rgba(255, 255, 85, 0.7)',
                            'rgba(170, 85, 255, 0.7)'
                        ],
                        borderColor: [
                            'rgba(255, 85, 85, 1)',
                            'rgba(255, 255, 85, 1)',
                            'rgba(170, 85, 255, 1)'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    cutout: '65%'
                }
            });

            // Simulate terminal updates
            const terminal = document.getElementById('terminal-content');
            let count = 0;
            setInterval(() => {
                if (count < 3) {
                    const newEntry = document.createElement('div');
                    newEntry.className = 'mb-1 text-gray-500';
                    newEntry.textContent = `[${new Date().toLocaleTimeString()}] Background scan: ${Math.floor(Math.random() * 100)} new items found`;
                    terminal.appendChild(newEntry);
                    terminal.scrollTop = terminal.scrollHeight;
                    count++;
                }
            }, 10000);

            // Blinking cursor effect
            setInterval(() => {
                const cursor = document.querySelector('.code-input');
                cursor.classList.toggle('opacity-0');
            }, 500);

            // Quick action functionality
            const quickActions = {
                currentScan: null,
                alertCount: 3,
                
                addAlert: function(type, message, details) {
                    const alertFeed = document.getElementById('alert-feed');
                    const alertCount = document.getElementById('alert-count');
                    const alertCountMain = document.getElementById('alert-count-main');
                    
                    const timestamp = new Date().toLocaleTimeString();
                    
                    const alertTypes = {
                        'CRITICAL': 'bg-red-900 text-red-300',
                        'WARNING': 'bg-yellow-900 text-yellow-300',
                        'SUSPICIOUS': 'bg-purple-900 text-purple-300',
                        'INFO': 'bg-blue-900 text-blue-300'
                    };
                    
                    const icons = {
                        'CRITICAL': 'fa-exclamation-triangle',
                        'WARNING': 'fa-exclamation-circle',
                        'SUSPICIOUS': 'fa-user-secret',
                        'INFO': 'fa-info-circle'
                    };
                    
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'p-3 border-b border-gray-800 hover:bg-gray-950 cursor-pointer transition';
                    alertDiv.innerHTML = `
                        <div class="flex justify-between items-start mb-1">
                            <span class="text-xs ${alertTypes[type]} px-2 py-0.5 rounded-full">${type}</span>
                            <span class="text-xs text-gray-500">${timestamp}</span>
                        </div>
                        <p class="text-sm">${message}</p>
                        <p class="text-xs text-gray-400 mt-1 flex items-center">
                            <i class="fas ${icons[type]} mr-1"></i> ${details}
                        </p>
                    `;
                    
                    alertFeed.insertBefore(alertDiv, alertFeed.firstChild);
                    this.alertCount++;
                    alertCount.textContent = `${this.alertCount} NEW`;
                    
                    // Update the main alert counter in analysis section
                    const currentCount = parseInt(alertCountMain.textContent.replace(/,/g, ''));
                    alertCountMain.textContent = (currentCount + 1).toLocaleString() + ' ALERTS';
                },
                
                terminalMessage: function(message) {
                    const terminal = document.getElementById('terminal-content');
                    const cursor = document.querySelector('.code-input');
                    cursor.remove();
                    
                    const newLine = document.createElement('div');
                    newLine.className = 'mb-1';
                    newLine.textContent = message;
                    terminal.appendChild(newLine);
                    
                    // Add back the cursor
                    const prompt = document.createElement('div');
                    prompt.className = 'mb-1';
                    prompt.innerHTML = '<span class="text-yellow-400">root@shadowwatch:~#</span><span class="code-input"> _</span>';
                    terminal.appendChild(prompt);
                    
                    terminal.scrollTop = terminal.scrollHeight;
                },
                
                runDeepScan: function() {
                    if (this.currentScan) return;
                    
                    this.currentScan = 'deep-scan';
                    document.getElementById('scan-progress').classList.remove('hidden');
                    document.getElementById('scan-result').classList.add('hidden');
                    
                    const statusMessage = document.getElementById('status-message');
                    statusMessage.classList.remove('text-yellow-400', 'blink');
                    statusMessage.classList.add('text-green-400');
                    statusMessage.textContent = 'RUNNING DEEP SCAN';
                    
                    this.terminalMessage('[+] Initiating deep scan protocol across darknet surfaces...');
                    
                    let progress = 0;
                    const scanInterval = setInterval(() => {
                        progress += Math.floor(Math.random() * 5) + 1;
                        if (progress >= 100) {
                            progress = 100;
                            clearInterval(scanInterval);
                            this.scanComplete();
                            this.terminalMessage('[+] Deep scan completed. Analyzing results...');
                            
                            // Add some findings
                            setTimeout(() => {
                                this.terminalMessage('[!] Detected 3 new data dumps on underground marketplaces');
                                this.terminalMessage('[!] Found credentials matching 14 corporate email domains');
                            }, 1500);
                            
                            // Add alert
                            setTimeout(() => {
                                this.addAlert('WARNING', 'Multiple credential dumps detected', '14 corporate domains affected, 3 marketplaces');
                            }, 3000);
                        }
                        
                        document.getElementById('progress-bar').style.width = `${progress}%`;
                        document.getElementById('progress-percent').textContent = `${progress}%`;
                    }, 150);
                },
                
                runCredentialTrace: function() {
                    if (this.currentScan) return;
                    
                    this.currentScan = 'credential-trace';
                    document.getElementById('scan-progress').classList.remove('hidden');
                    document.getElementById('scan-result').classList.add('hidden');
                    
                    const statusMessage = document.getElementById('status-message');
                    statusMessage.classList.remove('text-yellow-400', 'blink');
                    statusMessage.classList.add('text-green-400');
                    statusMessage.textContent = 'TRACING CREDENTIALS';
                    
                    this.terminalMessage('[+] Initializing credential tracing module...');
                    this.terminalMessage('[+] Checking known breaches and paste sites...');
                    
                    let progress = 0;
                    const scanInterval = setInterval(() => {
                        progress += Math.floor(Math.random() * 10) + 5;
                        if (progress >= 100) {
                            progress = 100;
                            clearInterval(scanInterval);
                            this.scanComplete();
                            this.terminalMessage('[+] Credential trace completed. Results available.');
                            
                            // Add some findings
                            setTimeout(() => {
                                this.terminalMessage('[!] Matched 143 credentials against corporate watchlist');
                                this.terminalMessage('[!] 12 CEO/C-level credentials found in recent dumps');
                            }, 1500);
                            
                            // Add alert
                            setTimeout(() => {
                                this.addAlert('CRITICAL', 'Executive credentials exposed', '12 C-level credentials found in recent breaches');
                            }, 3000);
                        }
                        
                        document.getElementById('progress-bar').style.width = `${progress}%`;
                        document.getElementById('progress-percent').textContent = `${progress}%`;
                    }, 100);
                },
                
                runChatterAnalysis: function() {
                    if (this.currentScan) return;
                    
                    this.currentScan = 'chatter-analysis';
                    document.getElementById('scan-progress').classList.remove('hidden');
                    document.getElementById('scan-result').classList.add('hidden');
                    
                    const statusMessage = document.getElementById('status-message');
                    statusMessage.classList.remove('text-yellow-400', 'blink');
                    statusMessage.classList.add('text-green-400');
                    statusMessage.textContent = 'ANALYZING CHATTER';
                    
                    this.terminalMessage('[+] Connecting to darknet chat channels...');
                    this.terminalMessage('[+] Initializing natural language processing...');
                    
                    let progress = 0;
                    const scanInterval = setInterval(() => {
                        progress += Math.floor(Math.random() * 15) + 5;
                        if (progress >= 100) {
                            progress = 100;
                            clearInterval(scanInterval);
                            this.scanComplete();
                            this.terminalMessage('[+] Chatter analysis completed. Trends identified.');
                            
                            // Add some findings
                            setTimeout(() => {
                                this.terminalMessage('[!] 72% increase in mentions of "supply chain attack"');
                                this.terminalMessage('[!] New ransomware variant "ShadowKill" discussed in 3 forums');
                            }, 1500);
                            
                            // Add alert
                            setTimeout(() => {
                                this.addAlert('SUSPICIOUS', 'Emerging ransomware variant detected', 'New "ShadowKill" ransomware discussed across 3 forums');
                            }, 3000);
                        }
                        
                        document.getElementById('progress-bar').style.width = `${progress}%`;
                        document.getElementById('progress-percent').textContent = `${progress}%`;
                    }, 80);
                },
                
                runIdentityAlert: function() {
                    if (this.currentScan) return;
                    
                    this.currentScan = 'identity-alert';
                    document.getElementById('scan-progress').classList.remove('hidden');
                    document.getElementById('scan-result').classList.add('hidden');
                    
                    const statusMessage = document.getElementById('status-message');
                    statusMessage.classList.remove('text-yellow-400', 'blink');
                    statusMessage.classList.add('text-green-400');
                    statusMessage.textContent = 'MONITORING IDENTITIES';
                    
                    this.terminalMessage('[+] Scanning for mentions of protected identities...');
                    this.terminalMessage('[+] Checking deep/dark web archives...');
                    
                    let progress = 0;
                    const scanInterval = setInterval(() => {
                        progress += Math.floor(Math.random() * 20) + 10;
                        if (progress >= 100) {
                            progress = 100;
                            clearInterval(scanInterval);
                            this.scanComplete();
                            this.terminalMessage('[+] Identity monitoring sweep completed.');
                            
                            // Add some findings
                            setTimeout(() => {
                                this.terminalMessage('[!] 3 protected identities mentioned in hacker forums');
                                this.terminalMessage('[!] 1 identity potentially compromised - immediate action advised');
                            }, 1500);
                            
                            // Add alert
                            setTimeout(() => {
                                this.addAlert('CRITICAL', 'Protected identity potentially exposed', 'High-profile identity compromise detected');
                            }, 3000);
                        }
                        
                        document.getElementById('progress-bar').style.width = `${progress}%`;
                        document.getElementById('progress-percent').textContent = `${progress}%`;
                    }, 50);
                },
                
                scanComplete: function() {
                    document.getElementById('progress-bar').style.width = '100%';
                    document.getElementById('progress-percent').textContent = '100%';
                    
                    setTimeout(() => {
                        document.getElementById('scan-result').classList.remove('hidden');
                        this.currentScan = null;
                        
                        const statusMessage = document.getElementById('status-message');
                        statusMessage.classList.remove('text-green-400');
                        statusMessage.classList.add('text-yellow-400', 'blink');
                        statusMessage.textContent = 'AWAITING COMMANDS';
                    }, 1000);
                },
                
                killswitch: function() {
                    const terminal = document.getElementById('terminal-content');
                    this.terminalMessage('[!] ACTIVATING EMERGENCY KILLSWITCH PROTOCOL...');
                    this.terminalMessage('[+] Disconnecting from TOR network...');
                    this.terminalMessage('[+] Wiping session data...');
                    this.terminalMessage('[+] Terminating all connections...');
                    this.terminalMessage('[!] SYSTEM SHUTDOWN INITIATED.');
                    
                    document.querySelectorAll('button').forEach(btn => {
                        btn.disabled = true;
                    });
                    
                    document.body.style.backgroundColor = '#000';
                    
                    setTimeout(() => {
                        document.body.innerHTML = `
                        <div class="h-screen w-full bg-black flex items-center justify-center">
                            <div class="text-center">
                                <h1 class="text-red-500 text-4xl mb-4">SYSTEM TERMINATED</h1>
                                <p class="text-gray-500">All connections terminated. Secure wipe complete.</p>
                                <p class="text-gray-500 mt-8">Refresh page to restart (simulated)</p>
                            </div>
                        </div>
                        `;
                    }, 3000);
                }
            };
            
            // Event listeners for quick actions
            document.getElementById('deep-scan').addEventListener('click', () => quickActions.runDeepScan());
            document.getElementById('credential-trace').addEventListener('click', () => quickActions.runCredentialTrace());
            document.getElementById('chatter-analysis').addEventListener('click', () => quickActions.runChatterAnalysis());
            document.getElementById('identity-alert').addEventListener('click', () => quickActions.runIdentityAlert());
            document.getElementById('killswitch').addEventListener('click', () => quickActions.killswitch());
            
            // Terminal control buttons
            document.getElementById('clear-terminal').addEventListener('click', () => {
                document.getElementById('terminal-content').innerHTML = '<div class="mb-1"><span class="text-yellow-400">root@shadowwatch:~#</span><span class="code-input"> _</span></div>';
            });
            
            document.getElementById('refresh-terminal').addEventListener('click', () => {
                quickActions.terminalMessage('[+] Refreshing terminal session...');
                setTimeout(() => {
                    quickActions.terminalMessage('[+] Terminal refreshed. All connections stable.');
                }, 800);
            });
        });
    </script>
</body>
</html>
```

# attached_assets\style.css

```css
body {
	padding: 2rem;
	font-family: -apple-system, BlinkMacSystemFont, "Arial", sans-serif;
}

h1 {
	font-size: 16px;
	margin-top: 0;
}

p {
	color: rgb(107, 114, 128);
	font-size: 15px;
	margin-bottom: 10px;
	margin-top: 5px;
}

.card {
	max-width: 620px;
	margin: 0 auto;
	padding: 16px;
	border: 1px solid lightgray;
	border-radius: 16px;
}

.card p:last-child {
	margin-bottom: 0;
}

```

# client\index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1" />
    <link rel="icon" type="image/ico" href="/favicon.ico" />
    <title>Synthians Cognitive Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

# client\public\favicon.ico

```ico
0000010001002020100000000000E80200001600000028000000200000004000
0000010004000000000080020000000000000000000000000000000000
0000000000000000000000000000000000000000000000800000800000
00808000800000008000800080800000C0C0C000808080000000FF0000
FF000000FFFF00FF000000FF00FF00FFFF0000FFFFFF00000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000222222220000000000000000000000
0000002222222222222000000000000000000000022222222222222220
0000000000000000002222222222222222220000000000000000222222
2222222222222200000000000000022222222222222222222220000000
0000002222222222222222222222000000000000222222222222222222
2222220000000000022222222222222222222222000000000002222222
2222222222222222000000000000222222222222222222222000000000
0000022222222222222222220000000000000000000222222222222000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000
00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

```

# client\src\App.tsx

```tsx
import React from "react";
import { Switch, Route } from "wouter";
import { Toaster } from "@/components/ui/toaster";
import { DashboardShell } from "./components/layout/DashboardShell";
import NotFound from "@/pages/not-found";
import Overview from "./pages/overview";
import MemoryCore from "./pages/memory-core";
import NeuralMemory from "./pages/neural-memory";
import CCE from "./pages/cce";
import AssembliesIndex from "./pages/assemblies/index";
import AssemblyDetail from "./pages/assemblies/[id]";
import LLMGuidance from "./pages/llm-guidance";
import Logs from "./pages/logs";
import Chat from "./pages/chat";
import Config from "./pages/config";
import Admin from "./pages/admin";
import { useEffect } from "react";
import { usePollingStore } from "./lib/store";
import { FeaturesProvider } from "./contexts/FeaturesContext";
import Phase59Tester from "./components/debug/Phase59Tester";

function Router() {
  const { startPolling, stopPolling } = usePollingStore();

  // Start the polling when the app loads
  useEffect(() => {
    startPolling();
    
    // Cleanup on unmount
    return () => {
      stopPolling();
    };
  }, [startPolling, stopPolling]);

  return (
    <DashboardShell>
      <Switch>
        <Route path="/" component={Overview} />
        <Route path="/memory-core" component={MemoryCore} />
        <Route path="/neural-memory" component={NeuralMemory} />
        <Route path="/cce" component={CCE} />
        <Route path="/assemblies" component={AssembliesIndex} />
        <Route path="/assemblies/:id" component={AssemblyDetail} />
        <Route path="/llm-guidance" component={LLMGuidance} />
        <Route path="/logs" component={Logs} />
        <Route path="/chat" component={Chat} />
        <Route path="/config" component={Config} />
        <Route path="/admin" component={Admin} />
        <Route path="/debug/phase59" component={Phase59Tester} />
        <Route component={NotFound} />
      </Switch>
    </DashboardShell>
  );
}

function App() {
  return (
    <FeaturesProvider>
      <Router />
      <Toaster />
    </FeaturesProvider>
  );
}

export default App;

```

# client\src\components\dashboard\ActivationExplanationView.tsx

```tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { ExplainActivationData } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface ActivationExplanationViewProps {
  activationData: ExplainActivationData | undefined;
  memoryId: string;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function ActivationExplanationView({ activationData, memoryId, isLoading, isError, error }: ActivationExplanationViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !activationData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load activation explanation data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate how close the similarity is to the threshold as a percentage
  const similarityPercentage = activationData.calculated_similarity != null && 
                               activationData.activation_threshold != null ? 
                               Math.min(
                                 100,
                                 Math.max(0, (activationData.calculated_similarity / activationData.activation_threshold) * 100)
                               ) : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Memory Activation Explanation</span>
          {activationData.passed_threshold ? (
            <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
              Activated
            </Badge>
          ) : (
            <Badge variant="secondary">
              Not Activated
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Memory ID</h3>
            <p className="font-mono">{memoryId}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Assembly ID</h3>
            <p className="font-mono">{activationData.assembly_id}</p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Check Time</h3>
            <p>{new Date(activationData.check_timestamp).toLocaleString()} ({formatTimeAgo(activationData.check_timestamp)})</p>
          </div>

          {activationData.trigger_context && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Trigger Context</h3>
              <p className="text-sm mt-1 bg-muted p-2 rounded whitespace-pre-wrap">
                {activationData.trigger_context}
              </p>
            </div>
          )}

          <div>
            <div className="flex justify-between mb-1">
              <h3 className="text-sm font-medium text-muted-foreground">Similarity Score</h3>
              <span className="text-sm">
                {activationData.calculated_similarity != null ? activationData.calculated_similarity.toFixed(4) : 'N/A'} / 
                {activationData.activation_threshold != null ? activationData.activation_threshold.toFixed(4) : 'N/A'}
              </span>
            </div>
            <Progress 
              value={similarityPercentage} 
              className="h-2 bg-muted"
            />
            <div 
              className={`h-1 mt-1 rounded-full ${activationData.passed_threshold ? 'bg-green-500' : 'bg-amber-500'}`} 
              style={{ width: `${similarityPercentage}%` }}
            ></div>
            <p className="text-xs mt-1 text-muted-foreground">
              {activationData.calculated_similarity != null && activationData.activation_threshold != null && (
                activationData.passed_threshold
                  ? `Exceeded threshold by ${((activationData.calculated_similarity / activationData.activation_threshold - 1) * 100).toFixed(1)}%`
                  : `${(100 - similarityPercentage).toFixed(1)}% below activation threshold`
              )}
            </p>
          </div>

          {activationData.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Notes</h3>
              <p className="text-sm italic">{activationData.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\AssemblyTable.tsx

```tsx
import React from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Link } from "wouter";

interface Assembly {
  id: string;
  name: string;
  member_count: number;
  updated_at: string;
  vector_index_updated_at?: string;
}

interface AssemblyTableProps {
  assemblies: Assembly[] | null;
  isLoading: boolean;
  title?: string;
  showFilters?: boolean;
}

export function AssemblyTable({ assemblies, isLoading, title = "Assemblies", showFilters = true }: AssemblyTableProps) {
  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    
    if (diffMin < 60) {
      return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
    } else if (diffMin < 1440) {
      const hours = Math.floor(diffMin / 60);
      return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    } else {
      const days = Math.floor(diffMin / 1440);
      return `${days} day${days === 1 ? '' : 's'} ago`;
    }
  };

  const getSyncStatus = (assembly: Assembly) => {
    if (!assembly.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-400",
        bgColor: "bg-muted/50"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-secondary",
        bgColor: "bg-muted/50"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-primary",
      bgColor: "bg-muted/50"
    };
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <div className="flex items-center">
          <CardTitle className="font-medium">{title}</CardTitle>
          <Badge variant="outline" className="ml-2 text-xs bg-muted/50 text-gray-300">Memory Core</Badge>
        </div>
        
        {showFilters && (
          <div className="flex space-x-2">
            <Button variant="ghost" size="icon" className="text-xs p-1 text-gray-400 hover:text-foreground">
              <i className="fas fa-filter"></i>
            </Button>
            <Button variant="ghost" size="icon" className="text-xs p-1 text-gray-400 hover:text-foreground">
              <i className="fas fa-sync-alt"></i>
            </Button>
          </div>
        )}
      </CardHeader>
      
      <div className="overflow-x-auto">
        <Table>
          <TableHeader className="bg-muted">
            <TableRow>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Assembly ID</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Name</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Member Count</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Updated</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Sync Status</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"></TableHead>
            </TableRow>
          </TableHeader>
          
          <TableBody className="divide-y divide-border">
            {isLoading ? (
              Array(5).fill(0).map((_, index) => (
                <TableRow key={index}>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-24" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-40" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-12" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-16" /></TableCell>
                  <TableCell className="px-4 py-2 text-right"><Skeleton className="h-4 w-10 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : assemblies && assemblies.length > 0 ? (
              assemblies.map((assembly) => {
                const syncStatus = getSyncStatus(assembly);
                return (
                  <TableRow key={assembly.id} className="hover:bg-muted">
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm font-mono text-secondary">
                      {assembly.id}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm">
                      {assembly.name}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm">
                      {assembly.member_count}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-xs text-gray-400">
                      {formatTimeAgo(assembly.updated_at)}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap">
                      <span className={`text-xs ${syncStatus.bgColor} ${syncStatus.color} px-2 py-0.5 rounded-full`}>
                        {syncStatus.label}
                      </span>
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-right text-sm font-medium">
                      <Link href={`/assemblies/${assembly.id}`}>
                        <a className="text-primary hover:text-accent text-xs">View</a>
                      </Link>
                    </TableCell>
                  </TableRow>
                );
              })
            ) : (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-4 text-gray-400">
                  No assemblies found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}

```

# client\src\components\dashboard\CCEChart.tsx

```tsx
import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface CCEChartProps {
  data: any[];
  isLoading: boolean;
  title: string;
}

export function CCEChart({ data, isLoading, title }: CCEChartProps) {
  // Prepare data for the stacked bar chart
  const prepareStackedData = (rawData: any[]) => {
    // Group data by hour
    const hourlyData: Record<string, { mac7b: number, mac13b: number, titan7b: number }> = {};
    
    if (!rawData || rawData.length === 0) {
      return [];
    }
    
    // Create empty hourly buckets for the last 12 hours
    const now = new Date();
    for (let i = 0; i < 12; i++) {
      const hour = new Date(now.getTime() - (i * 60 * 60 * 1000)).getHours();
      const hourLabel = `${hour}h`;
      hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
    }
    
    // Fill in the actual data
    rawData.forEach(response => {
      if (!response.variant_selection) return;
      
      const timestamp = new Date(response.timestamp);
      const hour = timestamp.getHours();
      const hourLabel = `${hour}h`;
      
      if (!hourlyData[hourLabel]) {
        hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
      }
      
      const variant = response.variant_selection.selected_variant.toLowerCase();
      if (variant.includes('mac-7b')) {
        hourlyData[hourLabel].mac7b += 1;
      } else if (variant.includes('mac-13b')) {
        hourlyData[hourLabel].mac13b += 1;
      } else if (variant.includes('titan')) {
        hourlyData[hourLabel].titan7b += 1;
      }
    });
    
    // Convert to array format for Recharts
    return Object.entries(hourlyData).map(([hour, counts]) => ({
      hour,
      'MAC-7b': counts.mac7b,
      'MAC-13b': counts.mac13b,
      'TITAN-7b': counts.titan7b
    }));
  };

  const chartData = prepareStackedData(data);
  
  // Calculate percentages for the legend
  const calculatePercentages = () => {
    if (!data || data.length === 0) return { mac7b: 0, mac13b: 0, titan7b: 0 };
    
    let mac7b = 0, mac13b = 0, titan7b = 0;
    let total = 0;
    
    data.forEach(response => {
      if (!response.variant_selection) return;
      
      const variant = response.variant_selection.selected_variant.toLowerCase();
      if (variant.includes('mac-7b')) {
        mac7b += 1;
      } else if (variant.includes('mac-13b')) {
        mac13b += 1;
      } else if (variant.includes('titan')) {
        titan7b += 1;
      }
      total += 1;
    });
    
    return {
      mac7b: total > 0 ? Math.round((mac7b / total) * 100) : 0,
      mac13b: total > 0 ? Math.round((mac13b / total) * 100) : 0,
      titan7b: total > 0 ? Math.round((titan7b / total) * 100) : 0
    };
  };
  
  const percentages = calculatePercentages();

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
        <button className="text-xs text-gray-400 hover:text-foreground">
          <i className="fas fa-expand-alt"></i>
        </button>
      </CardHeader>
      
      <CardContent className="p-4">
        {isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : (
          <div className="h-48 bg-muted rounded-md relative">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{
                  top: 20,
                  right: 10,
                  left: 10,
                  bottom: 20,
                }}
              >
                <XAxis 
                  dataKey="hour" 
                  tick={{ fontSize: 10, fill: '#666' }}
                  tickLine={{ stroke: '#333' }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fill: '#666' }}
                  tickLine={{ stroke: '#333' }}
                  axisLine={{ stroke: '#333' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1E1E1E', border: '1px solid #333' }}
                  labelStyle={{ color: '#ddd' }}
                />
                <Bar 
                  dataKey="MAC-7b" 
                  stackId="a" 
                  fill="#1EE4FF" 
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="MAC-13b" 
                  stackId="a" 
                  fill="#FF008C" 
                />
                <Bar 
                  dataKey="TITAN-7b" 
                  stackId="a" 
                  fill="#FF3EE8" 
                  radius={[0, 0, 4, 4]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        <div className="mt-4 flex items-center space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-secondary mr-1"></div>
            <span className="text-gray-400">MAC-7b ({percentages.mac7b}%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-primary mr-1"></div>
            <span className="text-gray-400">MAC-13b ({percentages.mac13b}%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-accent mr-1"></div>
            <span className="text-gray-400">TITAN-7b ({percentages.titan7b}%)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\DiagnosticAlerts.tsx

```tsx
import React from "react";
import { Link } from "wouter";
import { Alert } from "@shared/schema";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

interface DiagnosticAlertsProps {
  alerts: Alert[] | null;
  isLoading: boolean;
}

export function DiagnosticAlerts({ alerts, isLoading }: DiagnosticAlertsProps) {
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error':
        return 'fa-exclamation-circle';
      case 'warning':
        return 'fa-exclamation-triangle';
      case 'info':
      default:
        return 'fa-info-circle';
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'text-destructive';
      case 'warning':
        return 'text-primary';
      case 'info':
      default:
        return 'text-secondary';
    }
  };
  
  const getActionColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'text-destructive';
      case 'warning':
        return 'text-primary';
      case 'info':
      default:
        return 'text-secondary';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    
    if (diffMin < 60) {
      return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
    } else if (diffMin < 1440) {
      const hours = Math.floor(diffMin / 60);
      return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    } else {
      const days = Math.floor(diffMin / 1440);
      return `${days} day${days === 1 ? '' : 's'} ago`;
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Recent Diagnostic Alerts</h3>
        <Link href="/logs">
          <Button variant="outline" size="sm" className="text-xs">
            View All
          </Button>
        </Link>
      </div>
      
      <div className="space-y-3">
        {isLoading ? (
          Array(3).fill(0).map((_, index) => (
            <div key={index} className="p-3 bg-card rounded-lg border border-border">
              <div className="flex">
                <Skeleton className="h-5 w-5 rounded-full mr-3" />
                <div className="flex-1">
                  <Skeleton className="h-4 w-48 mb-2" />
                  <Skeleton className="h-3 w-60 mb-3" />
                  <div className="flex justify-between">
                    <Skeleton className="h-3 w-20" />
                    <Skeleton className="h-3 w-16" />
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : alerts && alerts.length > 0 ? (
          alerts.map((alert) => (
            <div 
              key={alert.id} 
              className="p-3 bg-card rounded-lg border border-border hover:border-primary flex items-start"
            >
              <div className={`${getAlertColor(alert.type)} mr-3 mt-0.5`}>
                <i className={`fas ${getAlertIcon(alert.type)}`}></i>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium mb-1">{alert.title}</h4>
                <p className="text-xs text-gray-400 mb-2">{alert.description}</p>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">{formatTimeAgo(alert.timestamp)}</span>
                  {alert.action && (
                    <button className={`text-xs ${getActionColor(alert.type)}`}>
                      {alert.action}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="p-4 text-center text-sm text-gray-400">
            <i className="fas fa-info-circle mr-2"></i>
            No diagnostic alerts to display
          </div>
        )}
      </div>
    </div>
  );
}

```

# client\src\components\dashboard\LineageView.tsx

```tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { LineageEntry } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface LineageViewProps {
  lineage: LineageEntry[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function LineageView({ lineage, isLoading, isError, error }: LineageViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !lineage) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load lineage data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (lineage.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center italic text-muted-foreground">
            <p>This assembly has no ancestry. It wasn't formed by a merge operation.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Assembly Lineage</span>
          <Badge variant="outline">{lineage.length} generations</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4 max-h-80 overflow-y-auto pr-2">
          {lineage.map((entry, index) => (
            <div key={entry.assembly_id} className="border rounded-md p-3 relative">
              {/* Connector lines */}
              {index < lineage.length - 1 && (
                <div className="absolute h-10 w-0.5 bg-border left-5 top-full"></div>
              )}
              
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold">
                    {entry.depth !== undefined && (
                      <Badge variant="outline" className="mr-2">Level {entry.depth}</Badge>
                    )}
                    {entry.name ? `${entry.name} (${entry.assembly_id})` : entry.assembly_id}
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Created: {entry.created_at ? formatTimeAgo(entry.created_at) : 'Unknown'}
                  </p>
                  {entry.status && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Status: <Badge variant="outline">{entry.status}</Badge>
                    </p>
                  )}
                </div>
                {entry.memory_count !== undefined && entry.memory_count !== null && (
                  <Badge variant="secondary">{entry.memory_count} memories</Badge>
                )}
              </div>
              
              {entry.parent_ids && entry.parent_ids.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-muted-foreground mb-1">Merged from:</p>
                  <div className="flex flex-wrap gap-1">
                    {entry.parent_ids.map((sourceId: string) => (
                      <Badge key={sourceId} variant="secondary" className="text-xs">
                        {sourceId}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\MergeExplanationView.tsx

```tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ExplainMergeData } from '@shared/schema';
import { formatDistanceToNow } from 'date-fns';

interface MergeExplanationViewProps {
  mergeData: ExplainMergeData | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function MergeExplanationView({ mergeData, isLoading, isError, error }: MergeExplanationViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Merge Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !mergeData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Merge Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load merge explanation data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Format the timestamp
  const formattedTime = mergeData.merge_timestamp ? 
    `${new Date(mergeData.merge_timestamp).toLocaleString()} (${formatDistanceToNow(new Date(mergeData.merge_timestamp), { addSuffix: true })})` : 
    'Unknown';

  return (
    <Card>
      <CardHeader>
        <CardTitle>Assembly Merge Explanation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Assembly ID</h3>
            <p className="font-mono">{mergeData.assembly_id}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Merge Timestamp</h3>
            <p>{formattedTime}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Similarity Threshold</h3>
            <p>{mergeData.similarity_threshold.toFixed(4)}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Source Assemblies</h3>
            <div className="flex flex-wrap gap-2 mt-1">
              {mergeData.source_assembly_ids.map(id => (
                <Badge key={id} variant="outline" className="font-mono">{id}</Badge>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Cleanup Status</h3>
            <div className="mt-1">
              {mergeData.cleanup_status === 'completed' && (
                <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                  Completed
                </Badge>
              )}
              {mergeData.cleanup_status === 'pending' && (
                <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                  Pending
                </Badge>
              )}
              {mergeData.cleanup_status === 'failed' && (
                <Badge variant="destructive">
                  Failed
                </Badge>
              )}
            </div>
          </div>

          {mergeData.cleanup_status === 'failed' && mergeData.error && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Error Details</h3>
              <p className="text-red-500 text-sm mt-1 font-mono bg-red-50 dark:bg-red-900/20 p-2 rounded">
                {mergeData.error}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\MergeLogView.tsx

```tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { MergeLogEntry } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface MergeLogViewProps {
  entries: MergeLogEntry[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function MergeLogView({ entries, isLoading, isError, error }: MergeLogViewProps) {
  if (isLoading) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !entries) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load merge log data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (entries.length === 0) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center italic text-muted-foreground">
            <p>No merge events have been recorded yet.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Group related events (merge and cleanup) by merge_event_id
  const groupedEntries: Record<string, MergeLogEntry[]> = {};
  entries.forEach(entry => {
    if (!groupedEntries[entry.merge_event_id]) {
      groupedEntries[entry.merge_event_id] = [];
    }
    groupedEntries[entry.merge_event_id].push(entry);
  });

  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Merge Activity Log</span>
          <Badge variant="outline">{entries.length} events</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b text-xs text-muted-foreground">
                <th className="text-left py-2 font-medium">Event ID</th>
                <th className="text-left py-2 font-medium">Type</th>
                <th className="text-left py-2 font-medium">Timestamp</th>
                <th className="text-left py-2 font-medium">Source Assemblies</th>
                <th className="text-left py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {Object.values(groupedEntries).map(group => {
                // Find the main merge event
                const mergeEvent = group.find(e => e.event_type === 'merge');
                if (!mergeEvent) return null;
                
                // Find the related cleanup event if it exists
                const cleanupEvent = group.find(e => e.event_type === 'cleanup_update');
                
                return (
                  <tr key={mergeEvent.merge_event_id} className="hover:bg-muted/50 text-sm">
                    <td className="py-3 font-mono">
                      {mergeEvent.merge_event_id.substring(0, 8)}...
                    </td>
                    <td className="py-3">
                      <Badge variant="outline">merge</Badge>
                    </td>
                    <td className="py-3">
                      {formatTimeAgo(mergeEvent.timestamp)}
                    </td>
                    <td className="py-3">
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {mergeEvent.source_assembly_ids?.map(id => (
                          <Badge key={id} variant="secondary" className="text-xs font-mono">
                            {id.substring(0, 6)}...
                          </Badge>
                        )) || 'N/A'}
                      </div>
                    </td>
                    <td className="py-3">
                      {!cleanupEvent && (
                        <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                          Pending
                        </Badge>
                      )}
                      {cleanupEvent && cleanupEvent.status === 'completed' && (
                        <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                          Completed
                        </Badge>
                      )}
                      {cleanupEvent && cleanupEvent.status === 'failed' && (
                        <Badge variant="destructive">
                          Failed
                          {cleanupEvent.error && (
                            <span className="ml-1 cursor-help" title={cleanupEvent.error}>ⓘ</span>
                          )}
                        </Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\MetricsChart.tsx

```tsx
import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface MetricsChartProps {
  title: string;
  data: any[];
  dataKeys: { key: string; color: string; name: string }[];
  isLoading: boolean;
  timeRange: string;
  onTimeRangeChange: (range: string) => void;
  summary?: { label: string; value: string | number; color?: string }[];
}

export function MetricsChart({
  title,
  data,
  dataKeys,
  isLoading,
  timeRange,
  onTimeRangeChange,
  summary
}: MetricsChartProps) {
  const timeRanges = ["24h", "12h", "6h", "1h"];
  
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
        
        <div className="flex space-x-2">
          {timeRanges.map((range) => (
            <button
              key={range}
              className={`text-xs px-2 py-1 rounded ${
                timeRange === range
                  ? "bg-muted text-secondary"
                  : "bg-muted/50 text-gray-300 hover:bg-muted"
              }`}
              onClick={() => onTimeRangeChange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent className="p-4">
        {isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : (
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{
                  top: 10,
                  right: 10,
                  left: 10,
                  bottom: 10,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#666" 
                  fontSize={10}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#1E1E1E", border: "1px solid #333" }}
                  labelStyle={{ color: "#ddd" }}
                />
                
                {dataKeys.map((dataKey) => (
                  <Line
                    key={dataKey.key}
                    type="monotone"
                    dataKey={dataKey.key}
                    name={dataKey.name}
                    stroke={dataKey.color}
                    activeDot={{ r: 4 }}
                    strokeWidth={2}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {summary && !isLoading && (
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            {summary.map((item, index) => (
              <div key={index} className="bg-muted p-2 rounded">
                <div className="text-gray-500">{item.label}</div>
                <div className={item.color || "text-primary"}>{item.value}</div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\OverviewCard.tsx

```tsx
import React from "react";
import { ServiceStatus } from "@shared/schema";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ServiceStatus as ServiceStatusComponent } from "../layout/ServiceStatus";
import { Skeleton } from "@/components/ui/skeleton";

interface OverviewCardProps {
  title: string;
  icon: string;
  service: ServiceStatus | null;
  metrics: Record<string, string | number> | null;
  isLoading: boolean;
}

export function OverviewCard({ title, icon, service, metrics, isLoading }: OverviewCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <div className="flex items-center">
          <i className={`fas fa-${icon} text-secondary mr-2`}></i>
          <CardTitle className="font-medium text-base">{title}</CardTitle>
        </div>
        {isLoading ? (
          <Skeleton className="w-16 h-5" />
        ) : service ? (
          <ServiceStatusComponent service={service} />
        ) : (
          <div className="text-xs text-destructive">
            <i className="fas fa-exclamation-circle mr-1"></i>
            <span>Unreachable</span>
          </div>
        )}
      </CardHeader>
      <CardContent className="p-4">
        {isLoading ? (
          <div className="grid grid-cols-2 gap-4 mb-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        ) : metrics ? (
          <div className="grid grid-cols-2 gap-4 mb-4">
            {Object.entries(metrics).map(([key, value], index) => (
              <div key={index} className="bg-muted p-3 rounded-md">
                <div className="text-xs text-gray-500 mb-1">{key}</div>
                <div className="text-lg font-mono">{value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-4 text-center text-sm text-gray-400">
            <i className="fas fa-exclamation-circle mr-2"></i>
            No metrics available
          </div>
        )}
        
        {service && (
          <div className="text-xs text-gray-400 flex justify-between">
            {service.uptime && <span>Uptime: {service.uptime}</span>}
            {service.version && <span>Version: {service.version}</span>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

```

# client\src\components\dashboard\SystemArchitecture.tsx

```tsx
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export function SystemArchitecture() {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border">
        <CardTitle className="font-medium text-base">System Architecture</CardTitle>
      </CardHeader>
      
      <CardContent className="p-6">
        <div className="relative h-64">
          {/* Memory Core Box */}
          <div className="absolute top-6 left-4 md:left-20 w-48 h-20 border border-secondary rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-secondary">Memory Core</div>
            <div className="text-xs text-gray-400 mt-1">Vector Database</div>
          </div>
          
          {/* Neural Memory Box */}
          <div className="absolute top-6 right-4 md:right-20 w-48 h-20 border border-primary rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-primary">Neural Memory</div>
            <div className="text-xs text-gray-400 mt-1">Emotional Loop</div>
          </div>
          
          {/* CCE Box */}
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 w-48 h-20 border border-accent rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-accent">CCE</div>
            <div className="text-xs text-gray-400 mt-1">Context Orchestration</div>
          </div>
          
          {/* Connection lines using SVG */}
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 240" preserveAspectRatio="none">
            {/* Left to Bottom */}
            <path 
              d="M120,70 L120,140 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#1EE4FF" 
              fill="none" 
            />
            
            {/* Right to Bottom */}
            <path 
              d="M480,70 L480,140 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FF008C" 
              fill="none" 
            />
            
            {/* Bottom to top */}
            <path 
              d="M300,180 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FF3EE8" 
              fill="none" 
            />
            
            {/* Left to Right */}
            <path 
              d="M168,40 C240,40 360,40 432,40" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FFFFFF" 
              fill="none" 
              opacity="0.3" 
            />
          </svg>
          
          {/* Connection labels */}
          <div className="absolute text-[10px] text-gray-500" style={{ left: '35%', top: '15%' }}>Memory Exchange</div>
          <div className="absolute text-[10px] text-gray-500" style={{ left: '20%', top: '40%' }}>Vector Queries</div>
          <div className="absolute text-[10px] text-gray-500" style={{ right: '20%', top: '40%' }}>Emotional Processing</div>
          <div className="absolute text-[10px] text-gray-500" style={{ left: '46%', top: '65%' }}>Context Flow</div>
        </div>
        
        <div className="flex justify-center mt-4 space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 border border-secondary rounded-sm mr-1"></div>
            <span className="text-gray-400">Storage</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 border border-primary rounded-sm mr-1"></div>
            <span className="text-gray-400">Processing</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 border border-accent rounded-sm mr-1"></div>
            <span className="text-gray-400">Orchestration</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

```

# client\src\components\debug\Phase59Tester.tsx

```tsx
import React, { useState } from 'react';
import { useFeatures } from '../../contexts/FeaturesContext';
import { useMergeLog, useRuntimeConfig, useAssemblyLineage } from '../../lib/api';
import { ReconciledMergeLogEntry, LineageEntry } from '@shared/schema';

/**
 * Debug component for testing Phase 5.9 features
 * Displays feature availability and sample data from Phase 5.9 endpoints
 */
const Phase59Tester: React.FC = () => {
  const { explainabilityEnabled, isLoading: featuresLoading } = useFeatures();
  const [testAssemblyId, setTestAssemblyId] = useState<string>(''); 
  
  // Test merge log endpoint
  const { data: mergeLogData, isLoading: mergeLogLoading, isError: mergeLogError } = useMergeLog(5); // Limit to 5 entries
  
  // Test runtime config endpoint
  const { data: configData, isLoading: configLoading, isError: configError } = useRuntimeConfig('memory-core');
  
  // Test lineage endpoint (conditionally enabled when assembly ID is entered)
  const { 
    data: lineageData, 
    isLoading: lineageLoading, 
    isError: lineageError 
  } = useAssemblyLineage(testAssemblyId || null);

  if (featuresLoading) {
    return <div className="p-4">Loading features...</div>;
  }

  return (
    <div className="p-4 space-y-6 border rounded-lg">
      <h2 className="text-xl font-bold">Phase 5.9 Features Debug</h2>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold">Feature Flags</h3>
        <p className="py-2">
          Explainability Enabled: <span className={explainabilityEnabled ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
            {explainabilityEnabled ? 'YES' : 'NO'}
          </span>
        </p>
        
        {!explainabilityEnabled && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 my-2">
            <p>Explainability features are disabled in the Memory Core configuration.</p>
            <p>Enable them by setting <code>ENABLE_EXPLAINABILITY=true</code> in the Memory Core service.</p>
          </div>
        )}
      </div>

      {explainabilityEnabled && (
        <>
          {/* Runtime Config Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Runtime Configuration</h3>
            {configLoading ? (
              <p>Loading configuration...</p>
            ) : configError ? (
              <p className="text-red-600">Error loading configuration</p>
            ) : (
              <pre className="bg-gray-800 text-green-400 p-4 rounded overflow-auto max-h-64">
                {JSON.stringify(configData, null, 2)}
              </pre>
            )}
          </div>
          
          {/* Merge Log Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Merge Log</h3>
            {mergeLogLoading ? (
              <p>Loading merge log...</p>
            ) : mergeLogError ? (
              <p className="text-red-600">Error loading merge log</p>
            ) : !mergeLogData?.reconciled_log_entries?.length ? (
              <p>No merge log entries available</p>
            ) : (
              <div className="overflow-auto max-h-64">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event ID</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {mergeLogData?.reconciled_log_entries?.map((entry: ReconciledMergeLogEntry, idx: number) => (
                      <tr key={idx}>
                        <td className="px-6 py-4 whitespace-nowrap">{new Date(entry.creation_timestamp).toLocaleString()}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{entry.merge_event_id?.substring(0, 8)}...</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            entry.final_cleanup_status === 'completed' ? 'bg-green-100 text-green-800' : 
                            entry.final_cleanup_status === 'failed' ? 'bg-red-100 text-red-800' : 
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {entry.final_cleanup_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          
          {/* Lineage Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Assembly Lineage Test</h3>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={testAssemblyId}
                onChange={(e) => setTestAssemblyId(e.target.value)}
                placeholder="Enter assembly ID"
                className="px-3 py-2 border rounded flex-grow"
              />
            </div>
            
            {testAssemblyId ? (
              lineageLoading ? (
                <p>Loading lineage...</p>
              ) : lineageError ? (
                <p className="text-red-600">Error loading lineage for assembly {testAssemblyId}</p>
              ) : !lineageData?.lineage?.length ? (
                <p>No lineage found for this assembly or assembly does not exist</p>
              ) : (
                <div className="overflow-auto max-h-64">
                  <h4 className="font-medium mb-2">Lineage Chain ({lineageData.lineage.length} entries):</h4>
                  <ul className="space-y-2">
                    {lineageData.lineage.map((entry: LineageEntry, idx: number) => (
                      <li key={idx} className="p-2 border rounded">
                        <div className="flex justify-between">
                          <span className="font-medium">{entry.assembly_id}</span>
                          <span className="text-sm text-gray-500">{entry.created_at ? new Date(entry.created_at).toLocaleString() : 'Unknown date'}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-gray-600">Status: </span>
                          {entry.status || 'Unknown'}
                        </div>
                        <div className="text-sm">
                          <span className="text-gray-600">Memories: </span>
                          {entry.memory_count || 'Unknown'}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )
            ) : (
              <p className="italic text-gray-500">Enter an assembly ID to test lineage lookup</p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Phase59Tester;

```

# client\src\components\layout\DashboardShell.tsx

```tsx
import React, { useState } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

interface DashboardShellProps {
  children: React.ReactNode;
}

export function DashboardShell({ children }: DashboardShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar - regular view */}
      <div className="hidden md:block">
        <Sidebar />
      </div>

      {/* Mobile Sidebar - shown when toggled */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div 
            className="fixed inset-0 bg-black/50" 
            onClick={toggleSidebar}
          ></div>
          <div className="fixed top-0 left-0 bottom-0 w-64 bg-sidebar border-r border-border">
            <Sidebar />
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        <TopBar toggleSidebar={toggleSidebar} />
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}

```

# client\src\components\layout\ServiceStatus.tsx

```tsx
import React from "react";
import { ServiceStatus as ServiceStatusType } from "@shared/schema";
import { cn } from "@/lib/utils";

interface ServiceStatusProps {
  service: ServiceStatusType;
}

export function ServiceStatus({ service }: ServiceStatusProps) {
  // Determine status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Healthy':
        return 'text-secondary';
      case 'Unhealthy':
      case 'Error':
        return 'text-destructive';
      case 'Checking...':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Healthy':
        return 'fa-check-circle';
      case 'Unhealthy':
      case 'Error':
        return 'fa-exclamation-circle';
      case 'Checking...':
        return 'fa-spinner fa-spin';
      default:
        return 'fa-question-circle';
    }
  };

  const statusColor = getStatusColor(service.status);
  const statusIcon = getStatusIcon(service.status);

  return (
    <div className="flex items-center">
      <div className={cn("w-2 h-2 rounded-full mr-1", {
        "bg-secondary pulse": service.status === 'Healthy',
        "bg-destructive pulse": service.status === 'Unhealthy' || service.status === 'Error',
        "bg-yellow-400 pulse": service.status === 'Checking...'
      })}></div>
      <span className={`text-xs ${statusColor}`}>
        <i className={`fas ${statusIcon} mr-1`}></i>
        {service.status}
      </span>
    </div>
  );
}

```

# client\src\components\layout\Sidebar.tsx

```tsx
import React from "react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";

type NavLinkProps = {
  href: string;
  icon: string;
  label: string;
  isActive: boolean;
};

const NavLink = ({ href, icon, label, isActive }: NavLinkProps) => {
  return (
    <Link href={href}>
      <div className={cn(
        "flex items-center px-4 py-2 text-sm hover:bg-muted hover:text-foreground cursor-pointer",
        isActive 
          ? "text-primary bg-muted border-l-2 border-primary" 
          : "text-gray-400"
      )}>
        <i className={`fas fa-${icon} w-5`}></i>
        <span>{label}</span>
      </div>
    </Link>
  );
};

const NavGroup = ({ title, children }: { title: string, children: React.ReactNode }) => {
  return (
    <>
      <div className="px-4 py-2 mt-4 text-xs text-gray-500 uppercase">{title}</div>
      {children}
    </>
  );
};

export function Sidebar() {
  const [location] = useLocation();

  return (
    <aside className="w-64 border-r border-border bg-sidebar hidden md:block">
      {/* Logo */}
      <div className="flex items-center p-4 border-b border-border">
        <div className="w-8 h-8 rounded-md bg-gradient-to-br from-primary to-accent flex items-center justify-center mr-2">
          <span className="text-white font-bold">S</span>
        </div>
        <div>
          <h1 className="text-lg font-bold text-primary">Synthians</h1>
          <p className="text-xs text-gray-500">Cognitive Dashboard v1.0</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="py-4">
        <NavGroup title="Monitoring">
          <NavLink 
            href="/" 
            icon="tachometer-alt" 
            label="System Overview" 
            isActive={location === "/"} 
          />
          <NavLink 
            href="/memory-core" 
            icon="database" 
            label="Memory Core" 
            isActive={location === "/memory-core"} 
          />
          <NavLink 
            href="/neural-memory" 
            icon="brain" 
            label="Neural Memory" 
            isActive={location === "/neural-memory"} 
          />
          <NavLink 
            href="/cce" 
            icon="sitemap" 
            label="CCE" 
            isActive={location === "/cce"} 
          />
        </NavGroup>

        <NavGroup title="Tools">
          <NavLink 
            href="/assemblies" 
            icon="puzzle-piece" 
            label="Assembly Inspector" 
            isActive={Boolean(location && location.indexOf("/assemblies") === 0)} 
          />
          <NavLink 
            href="/llm-guidance" 
            icon="comment" 
            label="LLM Guidance" 
            isActive={location === "/llm-guidance"} 
          />
          <NavLink 
            href="/logs" 
            icon="terminal" 
            label="Logs" 
            isActive={location === "/logs"} 
          />
          <NavLink 
            href="/chat" 
            icon="comments" 
            label="Chat Interface" 
            isActive={location === "/chat"} 
          />
        </NavGroup>

        <NavGroup title="Settings">
          <NavLink 
            href="/config" 
            icon="cog" 
            label="Configuration" 
            isActive={location === "/config"} 
          />
          <NavLink 
            href="/admin" 
            icon="wrench" 
            label="Admin Actions" 
            isActive={location === "/admin"} 
          />
        </NavGroup>
      </nav>
      
      {/* Status Indicator */}
      <div className="absolute bottom-0 w-64 p-4 border-t border-border">
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-secondary pulse mr-2"></div>
          <span className="text-xs text-gray-400">All Systems Operational</span>
        </div>
        <div className="mt-2 text-xs text-gray-500">Last updated: 1 minute ago</div>
      </div>
    </aside>
  );
}

```

# client\src\components\layout\TopBar.tsx

```tsx
import React from "react";
import { RefreshButton } from "../ui/RefreshButton";
import { usePollingStore } from "@/lib/store";
import { Link } from "wouter";

interface TopBarProps {
  toggleSidebar: () => void;
}

export function TopBar({ toggleSidebar }: TopBarProps) {
  const { pollingRate, setPollingRate, refreshAllData } = usePollingStore();

  const handleRefresh = () => {
    refreshAllData();
  };

  const handlePollingRateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPollingRate(parseInt(e.target.value));
  };

  return (
    <header className="bg-card border-b border-border px-4 py-3 flex justify-between items-center">
      <div className="flex items-center md:hidden">
        <button 
          onClick={toggleSidebar} 
          className="p-2 rounded-md bg-muted text-primary"
        >
          <i className="fas fa-bars"></i>
        </button>
        <div className="ml-3">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-primary to-accent flex items-center justify-center mr-2">
            <span className="text-white font-bold text-xs">S</span>
          </div>
          <h1 className="text-sm font-bold text-primary">Synthians</h1>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="relative max-w-xs w-64 hidden md:block">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center">
            <i className="fas fa-search text-gray-500"></i>
          </span>
          <input 
            type="text" 
            placeholder="Search..." 
            className="bg-muted text-sm rounded-md pl-10 pr-4 py-1.5 w-full focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        
        <RefreshButton onClick={handleRefresh} />
        
        <div className="w-px h-6 bg-muted mx-2"></div>
        
        <div className="text-xs text-gray-400 flex items-center">
          Poll rate: 
          <select 
            value={pollingRate} 
            onChange={handlePollingRateChange} 
            className="ml-2 bg-muted text-secondary p-1 rounded border border-border"
          >
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
            <option value={60000}>60s</option>
          </select>
        </div>
      </div>
      
      <div className="flex items-center">
        <span className="mr-2 text-xs text-gray-400 hidden md:inline-block">
          Memory Core: <span className="text-secondary">Healthy</span>
        </span>
        <Link href="/admin">
          <div className="text-xs px-2 py-1 rounded bg-muted border border-border text-gray-300 hover:bg-muted/90 cursor-pointer">
            <i className="fas fa-exclamation-triangle text-yellow-400 mr-1"></i>
            <span>Diagnostics</span>
          </div>
        </Link>
      </div>
    </header>
  );
}

```

# client\src\components\ui\accordion.tsx

```tsx
import * as React from "react"
import * as AccordionPrimitive from "@radix-ui/react-accordion"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

const Accordion = AccordionPrimitive.Root

const AccordionItem = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>
>(({ className, ...props }, ref) => (
  <AccordionPrimitive.Item
    ref={ref}
    className={cn("border-b", className)}
    {...props}
  />
))
AccordionItem.displayName = "AccordionItem"

const AccordionTrigger = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Header className="flex">
    <AccordionPrimitive.Trigger
      ref={ref}
      className={cn(
        "flex flex-1 items-center justify-between py-4 font-medium transition-all hover:underline [&[data-state=open]>svg]:rotate-180",
        className
      )}
      {...props}
    >
      {children}
      <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
))
AccordionTrigger.displayName = AccordionPrimitive.Trigger.displayName

const AccordionContent = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Content
    ref={ref}
    className="overflow-hidden text-sm transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
    {...props}
  >
    <div className={cn("pb-4 pt-0", className)}>{children}</div>
  </AccordionPrimitive.Content>
))

AccordionContent.displayName = AccordionPrimitive.Content.displayName

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent }

```

# client\src\components\ui\alert-dialog.tsx

```tsx
import * as React from "react"
import * as AlertDialogPrimitive from "@radix-ui/react-alert-dialog"

import { cn } from "@/lib/utils"
import { buttonVariants } from "@/components/ui/button"

const AlertDialog = AlertDialogPrimitive.Root

const AlertDialogTrigger = AlertDialogPrimitive.Trigger

const AlertDialogPortal = AlertDialogPrimitive.Portal

const AlertDialogOverlay = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Overlay
    className={cn(
      "fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
    ref={ref}
  />
))
AlertDialogOverlay.displayName = AlertDialogPrimitive.Overlay.displayName

const AlertDialogContent = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Content>
>(({ className, ...props }, ref) => (
  <AlertDialogPortal>
    <AlertDialogOverlay />
    <AlertDialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg",
        className
      )}
      {...props}
    />
  </AlertDialogPortal>
))
AlertDialogContent.displayName = AlertDialogPrimitive.Content.displayName

const AlertDialogHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col space-y-2 text-center sm:text-left",
      className
    )}
    {...props}
  />
)
AlertDialogHeader.displayName = "AlertDialogHeader"

const AlertDialogFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
      className
    )}
    {...props}
  />
)
AlertDialogFooter.displayName = "AlertDialogFooter"

const AlertDialogTitle = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Title
    ref={ref}
    className={cn("text-lg font-semibold", className)}
    {...props}
  />
))
AlertDialogTitle.displayName = AlertDialogPrimitive.Title.displayName

const AlertDialogDescription = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Description
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
AlertDialogDescription.displayName =
  AlertDialogPrimitive.Description.displayName

const AlertDialogAction = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Action>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Action>
>(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Action
    ref={ref}
    className={cn(buttonVariants(), className)}
    {...props}
  />
))
AlertDialogAction.displayName = AlertDialogPrimitive.Action.displayName

const AlertDialogCancel = React.forwardRef<
  React.ElementRef<typeof AlertDialogPrimitive.Cancel>,
  React.ComponentPropsWithoutRef<typeof AlertDialogPrimitive.Cancel>
>(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Cancel
    ref={ref}
    className={cn(
      buttonVariants({ variant: "outline" }),
      "mt-2 sm:mt-0",
      className
    )}
    {...props}
  />
))
AlertDialogCancel.displayName = AlertDialogPrimitive.Cancel.displayName

export {
  AlertDialog,
  AlertDialogPortal,
  AlertDialogOverlay,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogAction,
  AlertDialogCancel,
}

```

# client\src\components\ui\alert.tsx

```tsx
import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const alertVariants = cva(
  "relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground",
  {
    variants: {
      variant: {
        default: "bg-background text-foreground",
        destructive:
          "border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    role="alert"
    className={cn(alertVariants({ variant }), className)}
    {...props}
  />
))
Alert.displayName = "Alert"

const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
    {...props}
  />
))
AlertTitle.displayName = "AlertTitle"

const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm [&_p]:leading-relaxed", className)}
    {...props}
  />
))
AlertDescription.displayName = "AlertDescription"

export { Alert, AlertTitle, AlertDescription }

```

# client\src\components\ui\aspect-ratio.tsx

```tsx
import * as AspectRatioPrimitive from "@radix-ui/react-aspect-ratio"

const AspectRatio = AspectRatioPrimitive.Root

export { AspectRatio }

```

# client\src\components\ui\avatar.tsx

```tsx
import * as React from "react"
import * as AvatarPrimitive from "@radix-ui/react-avatar"

import { cn } from "@/lib/utils"

const Avatar = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Root>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full",
      className
    )}
    {...props}
  />
))
Avatar.displayName = AvatarPrimitive.Root.displayName

const AvatarImage = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Image>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Image>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Image
    ref={ref}
    className={cn("aspect-square h-full w-full", className)}
    {...props}
  />
))
AvatarImage.displayName = AvatarPrimitive.Image.displayName

const AvatarFallback = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Fallback>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Fallback>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Fallback
    ref={ref}
    className={cn(
      "flex h-full w-full items-center justify-center rounded-full bg-muted",
      className
    )}
    {...props}
  />
))
AvatarFallback.displayName = AvatarPrimitive.Fallback.displayName

export { Avatar, AvatarImage, AvatarFallback }

```

# client\src\components\ui\badge.tsx

```tsx
import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }

```

# client\src\components\ui\breadcrumb.tsx

```tsx
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { ChevronRight, MoreHorizontal } from "lucide-react"

import { cn } from "@/lib/utils"

const Breadcrumb = React.forwardRef<
  HTMLElement,
  React.ComponentPropsWithoutRef<"nav"> & {
    separator?: React.ReactNode
  }
>(({ ...props }, ref) => <nav ref={ref} aria-label="breadcrumb" {...props} />)
Breadcrumb.displayName = "Breadcrumb"

const BreadcrumbList = React.forwardRef<
  HTMLOListElement,
  React.ComponentPropsWithoutRef<"ol">
>(({ className, ...props }, ref) => (
  <ol
    ref={ref}
    className={cn(
      "flex flex-wrap items-center gap-1.5 break-words text-sm text-muted-foreground sm:gap-2.5",
      className
    )}
    {...props}
  />
))
BreadcrumbList.displayName = "BreadcrumbList"

const BreadcrumbItem = React.forwardRef<
  HTMLLIElement,
  React.ComponentPropsWithoutRef<"li">
>(({ className, ...props }, ref) => (
  <li
    ref={ref}
    className={cn("inline-flex items-center gap-1.5", className)}
    {...props}
  />
))
BreadcrumbItem.displayName = "BreadcrumbItem"

const BreadcrumbLink = React.forwardRef<
  HTMLAnchorElement,
  React.ComponentPropsWithoutRef<"a"> & {
    asChild?: boolean
  }
>(({ asChild, className, ...props }, ref) => {
  const Comp = asChild ? Slot : "a"

  return (
    <Comp
      ref={ref}
      className={cn("transition-colors hover:text-foreground", className)}
      {...props}
    />
  )
})
BreadcrumbLink.displayName = "BreadcrumbLink"

const BreadcrumbPage = React.forwardRef<
  HTMLSpanElement,
  React.ComponentPropsWithoutRef<"span">
>(({ className, ...props }, ref) => (
  <span
    ref={ref}
    role="link"
    aria-disabled="true"
    aria-current="page"
    className={cn("font-normal text-foreground", className)}
    {...props}
  />
))
BreadcrumbPage.displayName = "BreadcrumbPage"

const BreadcrumbSeparator = ({
  children,
  className,
  ...props
}: React.ComponentProps<"li">) => (
  <li
    role="presentation"
    aria-hidden="true"
    className={cn("[&>svg]:w-3.5 [&>svg]:h-3.5", className)}
    {...props}
  >
    {children ?? <ChevronRight />}
  </li>
)
BreadcrumbSeparator.displayName = "BreadcrumbSeparator"

const BreadcrumbEllipsis = ({
  className,
  ...props
}: React.ComponentProps<"span">) => (
  <span
    role="presentation"
    aria-hidden="true"
    className={cn("flex h-9 w-9 items-center justify-center", className)}
    {...props}
  >
    <MoreHorizontal className="h-4 w-4" />
    <span className="sr-only">More</span>
  </span>
)
BreadcrumbEllipsis.displayName = "BreadcrumbElipssis"

export {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
  BreadcrumbEllipsis,
}

```

# client\src\components\ui\button.tsx

```tsx
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }

```

# client\src\components\ui\calendar.tsx

```tsx
import * as React from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { DayPicker } from "react-day-picker"

import { cn } from "@/lib/utils"
import { buttonVariants } from "@/components/ui/button"

export type CalendarProps = React.ComponentProps<typeof DayPicker>

function Calendar({
  className,
  classNames,
  showOutsideDays = true,
  ...props
}: CalendarProps) {
  return (
    <DayPicker
      showOutsideDays={showOutsideDays}
      className={cn("p-3", className)}
      classNames={{
        months: "flex flex-col sm:flex-row space-y-4 sm:space-x-4 sm:space-y-0",
        month: "space-y-4",
        caption: "flex justify-center pt-1 relative items-center",
        caption_label: "text-sm font-medium",
        nav: "space-x-1 flex items-center",
        nav_button: cn(
          buttonVariants({ variant: "outline" }),
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
        ),
        nav_button_previous: "absolute left-1",
        nav_button_next: "absolute right-1",
        table: "w-full border-collapse space-y-1",
        head_row: "flex",
        head_cell:
          "text-muted-foreground rounded-md w-9 font-normal text-[0.8rem]",
        row: "flex w-full mt-2",
        cell: "h-9 w-9 text-center text-sm p-0 relative [&:has([aria-selected].day-range-end)]:rounded-r-md [&:has([aria-selected].day-outside)]:bg-accent/50 [&:has([aria-selected])]:bg-accent first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md focus-within:relative focus-within:z-20",
        day: cn(
          buttonVariants({ variant: "ghost" }),
          "h-9 w-9 p-0 font-normal aria-selected:opacity-100"
        ),
        day_range_end: "day-range-end",
        day_selected:
          "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
        day_today: "bg-accent text-accent-foreground",
        day_outside:
          "day-outside text-muted-foreground opacity-50 aria-selected:bg-accent/50 aria-selected:text-muted-foreground aria-selected:opacity-30",
        day_disabled: "text-muted-foreground opacity-50",
        day_range_middle:
          "aria-selected:bg-accent aria-selected:text-accent-foreground",
        day_hidden: "invisible",
        ...classNames,
      }}
      components={{
        IconLeft: ({ ...props }) => <ChevronLeft className="h-4 w-4" />,
        IconRight: ({ ...props }) => <ChevronRight className="h-4 w-4" />,
      }}
      {...props}
    />
  )
}
Calendar.displayName = "Calendar"

export { Calendar }

```

# client\src\components\ui\card.tsx

```tsx
import * as React from "react"

import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border bg-card text-card-foreground shadow-sm",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }

```

# client\src\components\ui\carousel.tsx

```tsx
import * as React from "react"
import useEmblaCarousel, {
  type UseEmblaCarouselType,
} from "embla-carousel-react"
import { ArrowLeft, ArrowRight } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

type CarouselApi = UseEmblaCarouselType[1]
type UseCarouselParameters = Parameters<typeof useEmblaCarousel>
type CarouselOptions = UseCarouselParameters[0]
type CarouselPlugin = UseCarouselParameters[1]

type CarouselProps = {
  opts?: CarouselOptions
  plugins?: CarouselPlugin
  orientation?: "horizontal" | "vertical"
  setApi?: (api: CarouselApi) => void
}

type CarouselContextProps = {
  carouselRef: ReturnType<typeof useEmblaCarousel>[0]
  api: ReturnType<typeof useEmblaCarousel>[1]
  scrollPrev: () => void
  scrollNext: () => void
  canScrollPrev: boolean
  canScrollNext: boolean
} & CarouselProps

const CarouselContext = React.createContext<CarouselContextProps | null>(null)

function useCarousel() {
  const context = React.useContext(CarouselContext)

  if (!context) {
    throw new Error("useCarousel must be used within a <Carousel />")
  }

  return context
}

const Carousel = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & CarouselProps
>(
  (
    {
      orientation = "horizontal",
      opts,
      setApi,
      plugins,
      className,
      children,
      ...props
    },
    ref
  ) => {
    const [carouselRef, api] = useEmblaCarousel(
      {
        ...opts,
        axis: orientation === "horizontal" ? "x" : "y",
      },
      plugins
    )
    const [canScrollPrev, setCanScrollPrev] = React.useState(false)
    const [canScrollNext, setCanScrollNext] = React.useState(false)

    const onSelect = React.useCallback((api: CarouselApi) => {
      if (!api) {
        return
      }

      setCanScrollPrev(api.canScrollPrev())
      setCanScrollNext(api.canScrollNext())
    }, [])

    const scrollPrev = React.useCallback(() => {
      api?.scrollPrev()
    }, [api])

    const scrollNext = React.useCallback(() => {
      api?.scrollNext()
    }, [api])

    const handleKeyDown = React.useCallback(
      (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault()
          scrollPrev()
        } else if (event.key === "ArrowRight") {
          event.preventDefault()
          scrollNext()
        }
      },
      [scrollPrev, scrollNext]
    )

    React.useEffect(() => {
      if (!api || !setApi) {
        return
      }

      setApi(api)
    }, [api, setApi])

    React.useEffect(() => {
      if (!api) {
        return
      }

      onSelect(api)
      api.on("reInit", onSelect)
      api.on("select", onSelect)

      return () => {
        api?.off("select", onSelect)
      }
    }, [api, onSelect])

    return (
      <CarouselContext.Provider
        value={{
          carouselRef,
          api: api,
          opts,
          orientation:
            orientation || (opts?.axis === "y" ? "vertical" : "horizontal"),
          scrollPrev,
          scrollNext,
          canScrollPrev,
          canScrollNext,
        }}
      >
        <div
          ref={ref}
          onKeyDownCapture={handleKeyDown}
          className={cn("relative", className)}
          role="region"
          aria-roledescription="carousel"
          {...props}
        >
          {children}
        </div>
      </CarouselContext.Provider>
    )
  }
)
Carousel.displayName = "Carousel"

const CarouselContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { carouselRef, orientation } = useCarousel()

  return (
    <div ref={carouselRef} className="overflow-hidden">
      <div
        ref={ref}
        className={cn(
          "flex",
          orientation === "horizontal" ? "-ml-4" : "-mt-4 flex-col",
          className
        )}
        {...props}
      />
    </div>
  )
})
CarouselContent.displayName = "CarouselContent"

const CarouselItem = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { orientation } = useCarousel()

  return (
    <div
      ref={ref}
      role="group"
      aria-roledescription="slide"
      className={cn(
        "min-w-0 shrink-0 grow-0 basis-full",
        orientation === "horizontal" ? "pl-4" : "pt-4",
        className
      )}
      {...props}
    />
  )
})
CarouselItem.displayName = "CarouselItem"

const CarouselPrevious = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<typeof Button>
>(({ className, variant = "outline", size = "icon", ...props }, ref) => {
  const { orientation, scrollPrev, canScrollPrev } = useCarousel()

  return (
    <Button
      ref={ref}
      variant={variant}
      size={size}
      className={cn(
        "absolute  h-8 w-8 rounded-full",
        orientation === "horizontal"
          ? "-left-12 top-1/2 -translate-y-1/2"
          : "-top-12 left-1/2 -translate-x-1/2 rotate-90",
        className
      )}
      disabled={!canScrollPrev}
      onClick={scrollPrev}
      {...props}
    >
      <ArrowLeft className="h-4 w-4" />
      <span className="sr-only">Previous slide</span>
    </Button>
  )
})
CarouselPrevious.displayName = "CarouselPrevious"

const CarouselNext = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<typeof Button>
>(({ className, variant = "outline", size = "icon", ...props }, ref) => {
  const { orientation, scrollNext, canScrollNext } = useCarousel()

  return (
    <Button
      ref={ref}
      variant={variant}
      size={size}
      className={cn(
        "absolute h-8 w-8 rounded-full",
        orientation === "horizontal"
          ? "-right-12 top-1/2 -translate-y-1/2"
          : "-bottom-12 left-1/2 -translate-x-1/2 rotate-90",
        className
      )}
      disabled={!canScrollNext}
      onClick={scrollNext}
      {...props}
    >
      <ArrowRight className="h-4 w-4" />
      <span className="sr-only">Next slide</span>
    </Button>
  )
})
CarouselNext.displayName = "CarouselNext"

export {
  type CarouselApi,
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
}

```

# client\src\components\ui\chart.tsx

```tsx
import * as React from "react"
import * as RechartsPrimitive from "recharts"

import { cn } from "@/lib/utils"

// Format: { THEME_NAME: CSS_SELECTOR }
const THEMES = { light: "", dark: ".dark" } as const

export type ChartConfig = {
  [k in string]: {
    label?: React.ReactNode
    icon?: React.ComponentType
  } & (
    | { color?: string; theme?: never }
    | { color?: never; theme: Record<keyof typeof THEMES, string> }
  )
}

type ChartContextProps = {
  config: ChartConfig
}

const ChartContext = React.createContext<ChartContextProps | null>(null)

function useChart() {
  const context = React.useContext(ChartContext)

  if (!context) {
    throw new Error("useChart must be used within a <ChartContainer />")
  }

  return context
}

const ChartContainer = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    config: ChartConfig
    children: React.ComponentProps<
      typeof RechartsPrimitive.ResponsiveContainer
    >["children"]
  }
>(({ id, className, children, config, ...props }, ref) => {
  const uniqueId = React.useId()
  const chartId = `chart-${id || uniqueId.replace(/:/g, "")}`

  return (
    <ChartContext.Provider value={{ config }}>
      <div
        data-chart={chartId}
        ref={ref}
        className={cn(
          "flex aspect-video justify-center text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line[stroke='#ccc']]:stroke-border/50 [&_.recharts-curve.recharts-tooltip-cursor]:stroke-border [&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-layer]:outline-none [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border [&_.recharts-radial-bar-background-sector]:fill-muted [&_.recharts-rectangle.recharts-tooltip-cursor]:fill-muted [&_.recharts-reference-line_[stroke='#ccc']]:stroke-border [&_.recharts-sector[stroke='#fff']]:stroke-transparent [&_.recharts-sector]:outline-none [&_.recharts-surface]:outline-none",
          className
        )}
        {...props}
      >
        <ChartStyle id={chartId} config={config} />
        <RechartsPrimitive.ResponsiveContainer>
          {children}
        </RechartsPrimitive.ResponsiveContainer>
      </div>
    </ChartContext.Provider>
  )
})
ChartContainer.displayName = "Chart"

const ChartStyle = ({ id, config }: { id: string; config: ChartConfig }) => {
  const colorConfig = Object.entries(config).filter(
    ([_, config]) => config.theme || config.color
  )

  if (!colorConfig.length) {
    return null
  }

  return (
    <style
      dangerouslySetInnerHTML={{
        __html: Object.entries(THEMES)
          .map(
            ([theme, prefix]) => `
${prefix} [data-chart=${id}] {
${colorConfig
  .map(([key, itemConfig]) => {
    const color =
      itemConfig.theme?.[theme as keyof typeof itemConfig.theme] ||
      itemConfig.color
    return color ? `  --color-${key}: ${color};` : null
  })
  .join("\n")}
}
`
          )
          .join("\n"),
      }}
    />
  )
}

const ChartTooltip = RechartsPrimitive.Tooltip

const ChartTooltipContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<typeof RechartsPrimitive.Tooltip> &
    React.ComponentProps<"div"> & {
      hideLabel?: boolean
      hideIndicator?: boolean
      indicator?: "line" | "dot" | "dashed"
      nameKey?: string
      labelKey?: string
    }
>(
  (
    {
      active,
      payload,
      className,
      indicator = "dot",
      hideLabel = false,
      hideIndicator = false,
      label,
      labelFormatter,
      labelClassName,
      formatter,
      color,
      nameKey,
      labelKey,
    },
    ref
  ) => {
    const { config } = useChart()

    const tooltipLabel = React.useMemo(() => {
      if (hideLabel || !payload?.length) {
        return null
      }

      const [item] = payload
      const key = `${labelKey || item.dataKey || item.name || "value"}`
      const itemConfig = getPayloadConfigFromPayload(config, item, key)
      const value =
        !labelKey && typeof label === "string"
          ? config[label as keyof typeof config]?.label || label
          : itemConfig?.label

      if (labelFormatter) {
        return (
          <div className={cn("font-medium", labelClassName)}>
            {labelFormatter(value, payload)}
          </div>
        )
      }

      if (!value) {
        return null
      }

      return <div className={cn("font-medium", labelClassName)}>{value}</div>
    }, [
      label,
      labelFormatter,
      payload,
      hideLabel,
      labelClassName,
      config,
      labelKey,
    ])

    if (!active || !payload?.length) {
      return null
    }

    const nestLabel = payload.length === 1 && indicator !== "dot"

    return (
      <div
        ref={ref}
        className={cn(
          "grid min-w-[8rem] items-start gap-1.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl",
          className
        )}
      >
        {!nestLabel ? tooltipLabel : null}
        <div className="grid gap-1.5">
          {payload.map((item, index) => {
            const key = `${nameKey || item.name || item.dataKey || "value"}`
            const itemConfig = getPayloadConfigFromPayload(config, item, key)
            const indicatorColor = color || item.payload.fill || item.color

            return (
              <div
                key={item.dataKey}
                className={cn(
                  "flex w-full flex-wrap items-stretch gap-2 [&>svg]:h-2.5 [&>svg]:w-2.5 [&>svg]:text-muted-foreground",
                  indicator === "dot" && "items-center"
                )}
              >
                {formatter && item?.value !== undefined && item.name ? (
                  formatter(item.value, item.name, item, index, item.payload)
                ) : (
                  <>
                    {itemConfig?.icon ? (
                      <itemConfig.icon />
                    ) : (
                      !hideIndicator && (
                        <div
                          className={cn(
                            "shrink-0 rounded-[2px] border-[--color-border] bg-[--color-bg]",
                            {
                              "h-2.5 w-2.5": indicator === "dot",
                              "w-1": indicator === "line",
                              "w-0 border-[1.5px] border-dashed bg-transparent":
                                indicator === "dashed",
                              "my-0.5": nestLabel && indicator === "dashed",
                            }
                          )}
                          style={
                            {
                              "--color-bg": indicatorColor,
                              "--color-border": indicatorColor,
                            } as React.CSSProperties
                          }
                        />
                      )
                    )}
                    <div
                      className={cn(
                        "flex flex-1 justify-between leading-none",
                        nestLabel ? "items-end" : "items-center"
                      )}
                    >
                      <div className="grid gap-1.5">
                        {nestLabel ? tooltipLabel : null}
                        <span className="text-muted-foreground">
                          {itemConfig?.label || item.name}
                        </span>
                      </div>
                      {item.value && (
                        <span className="font-mono font-medium tabular-nums text-foreground">
                          {item.value.toLocaleString()}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
      </div>
    )
  }
)
ChartTooltipContent.displayName = "ChartTooltip"

const ChartLegend = RechartsPrimitive.Legend

const ChartLegendContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> &
    Pick<RechartsPrimitive.LegendProps, "payload" | "verticalAlign"> & {
      hideIcon?: boolean
      nameKey?: string
    }
>(
  (
    { className, hideIcon = false, payload, verticalAlign = "bottom", nameKey },
    ref
  ) => {
    const { config } = useChart()

    if (!payload?.length) {
      return null
    }

    return (
      <div
        ref={ref}
        className={cn(
          "flex items-center justify-center gap-4",
          verticalAlign === "top" ? "pb-3" : "pt-3",
          className
        )}
      >
        {payload.map((item) => {
          const key = `${nameKey || item.dataKey || "value"}`
          const itemConfig = getPayloadConfigFromPayload(config, item, key)

          return (
            <div
              key={item.value}
              className={cn(
                "flex items-center gap-1.5 [&>svg]:h-3 [&>svg]:w-3 [&>svg]:text-muted-foreground"
              )}
            >
              {itemConfig?.icon && !hideIcon ? (
                <itemConfig.icon />
              ) : (
                <div
                  className="h-2 w-2 shrink-0 rounded-[2px]"
                  style={{
                    backgroundColor: item.color,
                  }}
                />
              )}
              {itemConfig?.label}
            </div>
          )
        })}
      </div>
    )
  }
)
ChartLegendContent.displayName = "ChartLegend"

// Helper to extract item config from a payload.
function getPayloadConfigFromPayload(
  config: ChartConfig,
  payload: unknown,
  key: string
) {
  if (typeof payload !== "object" || payload === null) {
    return undefined
  }

  const payloadPayload =
    "payload" in payload &&
    typeof payload.payload === "object" &&
    payload.payload !== null
      ? payload.payload
      : undefined

  let configLabelKey: string = key

  if (
    key in payload &&
    typeof payload[key as keyof typeof payload] === "string"
  ) {
    configLabelKey = payload[key as keyof typeof payload] as string
  } else if (
    payloadPayload &&
    key in payloadPayload &&
    typeof payloadPayload[key as keyof typeof payloadPayload] === "string"
  ) {
    configLabelKey = payloadPayload[
      key as keyof typeof payloadPayload
    ] as string
  }

  return configLabelKey in config
    ? config[configLabelKey]
    : config[key as keyof typeof config]
}

export {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  ChartStyle,
}

```

# client\src\components\ui\checkbox.tsx

```tsx
import * as React from "react"
import * as CheckboxPrimitive from "@radix-ui/react-checkbox"
import { Check } from "lucide-react"

import { cn } from "@/lib/utils"

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cn(
      "peer h-4 w-4 shrink-0 rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground",
      className
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator
      className={cn("flex items-center justify-center text-current")}
    >
      <Check className="h-4 w-4" />
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
))
Checkbox.displayName = CheckboxPrimitive.Root.displayName

export { Checkbox }

```

# client\src\components\ui\collapsible.tsx

```tsx
import * as CollapsiblePrimitive from "@radix-ui/react-collapsible"

const Collapsible = CollapsiblePrimitive.Root

const CollapsibleTrigger = CollapsiblePrimitive.CollapsibleTrigger

const CollapsibleContent = CollapsiblePrimitive.CollapsibleContent

export { Collapsible, CollapsibleTrigger, CollapsibleContent }

```

# client\src\components\ui\command.tsx

```tsx
import * as React from "react"
import { type DialogProps } from "@radix-ui/react-dialog"
import { Command as CommandPrimitive } from "cmdk"
import { Search } from "lucide-react"

import { cn } from "@/lib/utils"
import { Dialog, DialogContent } from "@/components/ui/dialog"

const Command = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive>
>(({ className, ...props }, ref) => (
  <CommandPrimitive
    ref={ref}
    className={cn(
      "flex h-full w-full flex-col overflow-hidden rounded-md bg-popover text-popover-foreground",
      className
    )}
    {...props}
  />
))
Command.displayName = CommandPrimitive.displayName

interface CommandDialogProps extends DialogProps {}

const CommandDialog = ({ children, ...props }: CommandDialogProps) => {
  return (
    <Dialog {...props}>
      <DialogContent className="overflow-hidden p-0 shadow-lg">
        <Command className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-muted-foreground [&_[cmdk-group]:not([hidden])_~[cmdk-group]]:pt-0 [&_[cmdk-group]]:px-2 [&_[cmdk-input-wrapper]_svg]:h-5 [&_[cmdk-input-wrapper]_svg]:w-5 [&_[cmdk-input]]:h-12 [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-3 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5">
          {children}
        </Command>
      </DialogContent>
    </Dialog>
  )
}

const CommandInput = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Input>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Input>
>(({ className, ...props }, ref) => (
  <div className="flex items-center border-b px-3" cmdk-input-wrapper="">
    <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
    <CommandPrimitive.Input
      ref={ref}
      className={cn(
        "flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  </div>
))

CommandInput.displayName = CommandPrimitive.Input.displayName

const CommandList = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.List>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.List
    ref={ref}
    className={cn("max-h-[300px] overflow-y-auto overflow-x-hidden", className)}
    {...props}
  />
))

CommandList.displayName = CommandPrimitive.List.displayName

const CommandEmpty = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Empty>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Empty>
>((props, ref) => (
  <CommandPrimitive.Empty
    ref={ref}
    className="py-6 text-center text-sm"
    {...props}
  />
))

CommandEmpty.displayName = CommandPrimitive.Empty.displayName

const CommandGroup = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Group>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Group>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Group
    ref={ref}
    className={cn(
      "overflow-hidden p-1 text-foreground [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-muted-foreground",
      className
    )}
    {...props}
  />
))

CommandGroup.displayName = CommandPrimitive.Group.displayName

const CommandSeparator = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 h-px bg-border", className)}
    {...props}
  />
))
CommandSeparator.displayName = CommandPrimitive.Separator.displayName

const CommandItem = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Item>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none data-[disabled=true]:pointer-events-none data-[selected='true']:bg-accent data-[selected=true]:text-accent-foreground data-[disabled=true]:opacity-50",
      className
    )}
    {...props}
  />
))

CommandItem.displayName = CommandPrimitive.Item.displayName

const CommandShortcut = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) => {
  return (
    <span
      className={cn(
        "ml-auto text-xs tracking-widest text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}
CommandShortcut.displayName = "CommandShortcut"

export {
  Command,
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandShortcut,
  CommandSeparator,
}

```

# client\src\components\ui\context-menu.tsx

```tsx
import * as React from "react"
import * as ContextMenuPrimitive from "@radix-ui/react-context-menu"
import { Check, ChevronRight, Circle } from "lucide-react"

import { cn } from "@/lib/utils"

const ContextMenu = ContextMenuPrimitive.Root

const ContextMenuTrigger = ContextMenuPrimitive.Trigger

const ContextMenuGroup = ContextMenuPrimitive.Group

const ContextMenuPortal = ContextMenuPrimitive.Portal

const ContextMenuSub = ContextMenuPrimitive.Sub

const ContextMenuRadioGroup = ContextMenuPrimitive.RadioGroup

const ContextMenuSubTrigger = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.SubTrigger>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.SubTrigger> & {
    inset?: boolean
  }
>(({ className, inset, children, ...props }, ref) => (
  <ContextMenuPrimitive.SubTrigger
    ref={ref}
    className={cn(
      "flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[state=open]:bg-accent data-[state=open]:text-accent-foreground",
      inset && "pl-8",
      className
    )}
    {...props}
  >
    {children}
    <ChevronRight className="ml-auto h-4 w-4" />
  </ContextMenuPrimitive.SubTrigger>
))
ContextMenuSubTrigger.displayName = ContextMenuPrimitive.SubTrigger.displayName

const ContextMenuSubContent = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.SubContent>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.SubContent>
>(({ className, ...props }, ref) => (
  <ContextMenuPrimitive.SubContent
    ref={ref}
    className={cn(
      "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))
ContextMenuSubContent.displayName = ContextMenuPrimitive.SubContent.displayName

const ContextMenuContent = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.Content>
>(({ className, ...props }, ref) => (
  <ContextMenuPrimitive.Portal>
    <ContextMenuPrimitive.Content
      ref={ref}
      className={cn(
        "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md animate-in fade-in-80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        className
      )}
      {...props}
    />
  </ContextMenuPrimitive.Portal>
))
ContextMenuContent.displayName = ContextMenuPrimitive.Content.displayName

const ContextMenuItem = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.Item> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <ContextMenuPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
ContextMenuItem.displayName = ContextMenuPrimitive.Item.displayName

const ContextMenuCheckboxItem = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.CheckboxItem>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.CheckboxItem>
>(({ className, children, checked, ...props }, ref) => (
  <ContextMenuPrimitive.CheckboxItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    checked={checked}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <ContextMenuPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </ContextMenuPrimitive.ItemIndicator>
    </span>
    {children}
  </ContextMenuPrimitive.CheckboxItem>
))
ContextMenuCheckboxItem.displayName =
  ContextMenuPrimitive.CheckboxItem.displayName

const ContextMenuRadioItem = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.RadioItem>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.RadioItem>
>(({ className, children, ...props }, ref) => (
  <ContextMenuPrimitive.RadioItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <ContextMenuPrimitive.ItemIndicator>
        <Circle className="h-2 w-2 fill-current" />
      </ContextMenuPrimitive.ItemIndicator>
    </span>
    {children}
  </ContextMenuPrimitive.RadioItem>
))
ContextMenuRadioItem.displayName = ContextMenuPrimitive.RadioItem.displayName

const ContextMenuLabel = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.Label> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <ContextMenuPrimitive.Label
    ref={ref}
    className={cn(
      "px-2 py-1.5 text-sm font-semibold text-foreground",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
ContextMenuLabel.displayName = ContextMenuPrimitive.Label.displayName

const ContextMenuSeparator = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof ContextMenuPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <ContextMenuPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-border", className)}
    {...props}
  />
))
ContextMenuSeparator.displayName = ContextMenuPrimitive.Separator.displayName

const ContextMenuShortcut = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) => {
  return (
    <span
      className={cn(
        "ml-auto text-xs tracking-widest text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}
ContextMenuShortcut.displayName = "ContextMenuShortcut"

export {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuCheckboxItem,
  ContextMenuRadioItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuShortcut,
  ContextMenuGroup,
  ContextMenuPortal,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuRadioGroup,
}

```

# client\src\components\ui\dialog.tsx

```tsx
import * as React from "react"
import * as DialogPrimitive from "@radix-ui/react-dialog"
import { X } from "lucide-react"

import { cn } from "@/lib/utils"

const Dialog = DialogPrimitive.Root

const DialogTrigger = DialogPrimitive.Trigger

const DialogPortal = DialogPrimitive.Portal

const DialogClose = DialogPrimitive.Close

const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      "fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
  />
))
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg",
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
        <X className="h-4 w-4" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName

const DialogHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col space-y-1.5 text-center sm:text-left",
      className
    )}
    {...props}
  />
)
DialogHeader.displayName = "DialogHeader"

const DialogFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
      className
    )}
    {...props}
  />
)
DialogFooter.displayName = "DialogFooter"

const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
DialogTitle.displayName = DialogPrimitive.Title.displayName

const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
DialogDescription.displayName = DialogPrimitive.Description.displayName

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogClose,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
}

```

# client\src\components\ui\drawer.tsx

```tsx
import * as React from "react"
import { Drawer as DrawerPrimitive } from "vaul"

import { cn } from "@/lib/utils"

const Drawer = ({
  shouldScaleBackground = true,
  ...props
}: React.ComponentProps<typeof DrawerPrimitive.Root>) => (
  <DrawerPrimitive.Root
    shouldScaleBackground={shouldScaleBackground}
    {...props}
  />
)
Drawer.displayName = "Drawer"

const DrawerTrigger = DrawerPrimitive.Trigger

const DrawerPortal = DrawerPrimitive.Portal

const DrawerClose = DrawerPrimitive.Close

const DrawerOverlay = React.forwardRef<
  React.ElementRef<typeof DrawerPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DrawerPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DrawerPrimitive.Overlay
    ref={ref}
    className={cn("fixed inset-0 z-50 bg-black/80", className)}
    {...props}
  />
))
DrawerOverlay.displayName = DrawerPrimitive.Overlay.displayName

const DrawerContent = React.forwardRef<
  React.ElementRef<typeof DrawerPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DrawerPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DrawerPortal>
    <DrawerOverlay />
    <DrawerPrimitive.Content
      ref={ref}
      className={cn(
        "fixed inset-x-0 bottom-0 z-50 mt-24 flex h-auto flex-col rounded-t-[10px] border bg-background",
        className
      )}
      {...props}
    >
      <div className="mx-auto mt-4 h-2 w-[100px] rounded-full bg-muted" />
      {children}
    </DrawerPrimitive.Content>
  </DrawerPortal>
))
DrawerContent.displayName = "DrawerContent"

const DrawerHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn("grid gap-1.5 p-4 text-center sm:text-left", className)}
    {...props}
  />
)
DrawerHeader.displayName = "DrawerHeader"

const DrawerFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn("mt-auto flex flex-col gap-2 p-4", className)}
    {...props}
  />
)
DrawerFooter.displayName = "DrawerFooter"

const DrawerTitle = React.forwardRef<
  React.ElementRef<typeof DrawerPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DrawerPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DrawerPrimitive.Title
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
DrawerTitle.displayName = DrawerPrimitive.Title.displayName

const DrawerDescription = React.forwardRef<
  React.ElementRef<typeof DrawerPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DrawerPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DrawerPrimitive.Description
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
DrawerDescription.displayName = DrawerPrimitive.Description.displayName

export {
  Drawer,
  DrawerPortal,
  DrawerOverlay,
  DrawerTrigger,
  DrawerClose,
  DrawerContent,
  DrawerHeader,
  DrawerFooter,
  DrawerTitle,
  DrawerDescription,
}

```

# client\src\components\ui\dropdown-menu.tsx

```tsx
import * as React from "react"
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu"
import { Check, ChevronRight, Circle } from "lucide-react"

import { cn } from "@/lib/utils"

const DropdownMenu = DropdownMenuPrimitive.Root

const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger

const DropdownMenuGroup = DropdownMenuPrimitive.Group

const DropdownMenuPortal = DropdownMenuPrimitive.Portal

const DropdownMenuSub = DropdownMenuPrimitive.Sub

const DropdownMenuRadioGroup = DropdownMenuPrimitive.RadioGroup

const DropdownMenuSubTrigger = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.SubTrigger>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.SubTrigger> & {
    inset?: boolean
  }
>(({ className, inset, children, ...props }, ref) => (
  <DropdownMenuPrimitive.SubTrigger
    ref={ref}
    className={cn(
      "flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent data-[state=open]:bg-accent",
      inset && "pl-8",
      className
    )}
    {...props}
  >
    {children}
    <ChevronRight className="ml-auto h-4 w-4" />
  </DropdownMenuPrimitive.SubTrigger>
))
DropdownMenuSubTrigger.displayName =
  DropdownMenuPrimitive.SubTrigger.displayName

const DropdownMenuSubContent = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.SubContent>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.SubContent>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.SubContent
    ref={ref}
    className={cn(
      "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))
DropdownMenuSubContent.displayName =
  DropdownMenuPrimitive.SubContent.displayName

const DropdownMenuContent = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        className
      )}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
))
DropdownMenuContent.displayName = DropdownMenuPrimitive.Content.displayName

const DropdownMenuItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <DropdownMenuPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
DropdownMenuItem.displayName = DropdownMenuPrimitive.Item.displayName

const DropdownMenuCheckboxItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.CheckboxItem>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.CheckboxItem>
>(({ className, children, checked, ...props }, ref) => (
  <DropdownMenuPrimitive.CheckboxItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    checked={checked}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <DropdownMenuPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </DropdownMenuPrimitive.ItemIndicator>
    </span>
    {children}
  </DropdownMenuPrimitive.CheckboxItem>
))
DropdownMenuCheckboxItem.displayName =
  DropdownMenuPrimitive.CheckboxItem.displayName

const DropdownMenuRadioItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.RadioItem>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.RadioItem>
>(({ className, children, ...props }, ref) => (
  <DropdownMenuPrimitive.RadioItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <DropdownMenuPrimitive.ItemIndicator>
        <Circle className="h-2 w-2 fill-current" />
      </DropdownMenuPrimitive.ItemIndicator>
    </span>
    {children}
  </DropdownMenuPrimitive.RadioItem>
))
DropdownMenuRadioItem.displayName = DropdownMenuPrimitive.RadioItem.displayName

const DropdownMenuLabel = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Label> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <DropdownMenuPrimitive.Label
    ref={ref}
    className={cn(
      "px-2 py-1.5 text-sm font-semibold",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
DropdownMenuLabel.displayName = DropdownMenuPrimitive.Label.displayName

const DropdownMenuSeparator = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
))
DropdownMenuSeparator.displayName = DropdownMenuPrimitive.Separator.displayName

const DropdownMenuShortcut = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) => {
  return (
    <span
      className={cn("ml-auto text-xs tracking-widest opacity-60", className)}
      {...props}
    />
  )
}
DropdownMenuShortcut.displayName = "DropdownMenuShortcut"

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuGroup,
  DropdownMenuPortal,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuRadioGroup,
}

```

# client\src\components\ui\form.tsx

```tsx
import * as React from "react"
import * as LabelPrimitive from "@radix-ui/react-label"
import { Slot } from "@radix-ui/react-slot"
import {
  Controller,
  ControllerProps,
  FieldPath,
  FieldValues,
  FormProvider,
  useFormContext,
} from "react-hook-form"

import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"

const Form = FormProvider

type FormFieldContextValue<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
> = {
  name: TName
}

const FormFieldContext = React.createContext<FormFieldContextValue>(
  {} as FormFieldContextValue
)

const FormField = <
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
>({
  ...props
}: ControllerProps<TFieldValues, TName>) => {
  return (
    <FormFieldContext.Provider value={{ name: props.name }}>
      <Controller {...props} />
    </FormFieldContext.Provider>
  )
}

const useFormField = () => {
  const fieldContext = React.useContext(FormFieldContext)
  const itemContext = React.useContext(FormItemContext)
  const { getFieldState, formState } = useFormContext()

  const fieldState = getFieldState(fieldContext.name, formState)

  if (!fieldContext) {
    throw new Error("useFormField should be used within <FormField>")
  }

  const { id } = itemContext

  return {
    id,
    name: fieldContext.name,
    formItemId: `${id}-form-item`,
    formDescriptionId: `${id}-form-item-description`,
    formMessageId: `${id}-form-item-message`,
    ...fieldState,
  }
}

type FormItemContextValue = {
  id: string
}

const FormItemContext = React.createContext<FormItemContextValue>(
  {} as FormItemContextValue
)

const FormItem = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const id = React.useId()

  return (
    <FormItemContext.Provider value={{ id }}>
      <div ref={ref} className={cn("space-y-2", className)} {...props} />
    </FormItemContext.Provider>
  )
})
FormItem.displayName = "FormItem"

const FormLabel = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root>
>(({ className, ...props }, ref) => {
  const { error, formItemId } = useFormField()

  return (
    <Label
      ref={ref}
      className={cn(error && "text-destructive", className)}
      htmlFor={formItemId}
      {...props}
    />
  )
})
FormLabel.displayName = "FormLabel"

const FormControl = React.forwardRef<
  React.ElementRef<typeof Slot>,
  React.ComponentPropsWithoutRef<typeof Slot>
>(({ ...props }, ref) => {
  const { error, formItemId, formDescriptionId, formMessageId } = useFormField()

  return (
    <Slot
      ref={ref}
      id={formItemId}
      aria-describedby={
        !error
          ? `${formDescriptionId}`
          : `${formDescriptionId} ${formMessageId}`
      }
      aria-invalid={!!error}
      {...props}
    />
  )
})
FormControl.displayName = "FormControl"

const FormDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => {
  const { formDescriptionId } = useFormField()

  return (
    <p
      ref={ref}
      id={formDescriptionId}
      className={cn("text-sm text-muted-foreground", className)}
      {...props}
    />
  )
})
FormDescription.displayName = "FormDescription"

const FormMessage = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, children, ...props }, ref) => {
  const { error, formMessageId } = useFormField()
  const body = error ? String(error?.message) : children

  if (!body) {
    return null
  }

  return (
    <p
      ref={ref}
      id={formMessageId}
      className={cn("text-sm font-medium text-destructive", className)}
      {...props}
    >
      {body}
    </p>
  )
})
FormMessage.displayName = "FormMessage"

export {
  useFormField,
  Form,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  FormMessage,
  FormField,
}

```

# client\src\components\ui\hover-card.tsx

```tsx
import * as React from "react"
import * as HoverCardPrimitive from "@radix-ui/react-hover-card"

import { cn } from "@/lib/utils"

const HoverCard = HoverCardPrimitive.Root

const HoverCardTrigger = HoverCardPrimitive.Trigger

const HoverCardContent = React.forwardRef<
  React.ElementRef<typeof HoverCardPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof HoverCardPrimitive.Content>
>(({ className, align = "center", sideOffset = 4, ...props }, ref) => (
  <HoverCardPrimitive.Content
    ref={ref}
    align={align}
    sideOffset={sideOffset}
    className={cn(
      "z-50 w-64 rounded-md border bg-popover p-4 text-popover-foreground shadow-md outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))
HoverCardContent.displayName = HoverCardPrimitive.Content.displayName

export { HoverCard, HoverCardTrigger, HoverCardContent }

```

# client\src\components\ui\input-otp.tsx

```tsx
import * as React from "react"
import { OTPInput, OTPInputContext } from "input-otp"
import { Dot } from "lucide-react"

import { cn } from "@/lib/utils"

const InputOTP = React.forwardRef<
  React.ElementRef<typeof OTPInput>,
  React.ComponentPropsWithoutRef<typeof OTPInput>
>(({ className, containerClassName, ...props }, ref) => (
  <OTPInput
    ref={ref}
    containerClassName={cn(
      "flex items-center gap-2 has-[:disabled]:opacity-50",
      containerClassName
    )}
    className={cn("disabled:cursor-not-allowed", className)}
    {...props}
  />
))
InputOTP.displayName = "InputOTP"

const InputOTPGroup = React.forwardRef<
  React.ElementRef<"div">,
  React.ComponentPropsWithoutRef<"div">
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center", className)} {...props} />
))
InputOTPGroup.displayName = "InputOTPGroup"

const InputOTPSlot = React.forwardRef<
  React.ElementRef<"div">,
  React.ComponentPropsWithoutRef<"div"> & { index: number }
>(({ index, className, ...props }, ref) => {
  const inputOTPContext = React.useContext(OTPInputContext)
  const { char, hasFakeCaret, isActive } = inputOTPContext.slots[index]

  return (
    <div
      ref={ref}
      className={cn(
        "relative flex h-10 w-10 items-center justify-center border-y border-r border-input text-sm transition-all first:rounded-l-md first:border-l last:rounded-r-md",
        isActive && "z-10 ring-2 ring-ring ring-offset-background",
        className
      )}
      {...props}
    >
      {char}
      {hasFakeCaret && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="h-4 w-px animate-caret-blink bg-foreground duration-1000" />
        </div>
      )}
    </div>
  )
})
InputOTPSlot.displayName = "InputOTPSlot"

const InputOTPSeparator = React.forwardRef<
  React.ElementRef<"div">,
  React.ComponentPropsWithoutRef<"div">
>(({ ...props }, ref) => (
  <div ref={ref} role="separator" {...props}>
    <Dot />
  </div>
))
InputOTPSeparator.displayName = "InputOTPSeparator"

export { InputOTP, InputOTPGroup, InputOTPSlot, InputOTPSeparator }

```

# client\src\components\ui\input.tsx

```tsx
import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }

```

# client\src\components\ui\label.tsx

```tsx
import * as React from "react"
import * as LabelPrimitive from "@radix-ui/react-label"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const labelVariants = cva(
  "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
)

const Label = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root> &
    VariantProps<typeof labelVariants>
>(({ className, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    className={cn(labelVariants(), className)}
    {...props}
  />
))
Label.displayName = LabelPrimitive.Root.displayName

export { Label }

```

# client\src\components\ui\menubar.tsx

```tsx
import * as React from "react"
import * as MenubarPrimitive from "@radix-ui/react-menubar"
import { Check, ChevronRight, Circle } from "lucide-react"

import { cn } from "@/lib/utils"

const MenubarMenu = MenubarPrimitive.Menu

const MenubarGroup = MenubarPrimitive.Group

const MenubarPortal = MenubarPrimitive.Portal

const MenubarSub = MenubarPrimitive.Sub

const MenubarRadioGroup = MenubarPrimitive.RadioGroup

const Menubar = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Root>
>(({ className, ...props }, ref) => (
  <MenubarPrimitive.Root
    ref={ref}
    className={cn(
      "flex h-10 items-center space-x-1 rounded-md border bg-background p-1",
      className
    )}
    {...props}
  />
))
Menubar.displayName = MenubarPrimitive.Root.displayName

const MenubarTrigger = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <MenubarPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex cursor-default select-none items-center rounded-sm px-3 py-1.5 text-sm font-medium outline-none focus:bg-accent focus:text-accent-foreground data-[state=open]:bg-accent data-[state=open]:text-accent-foreground",
      className
    )}
    {...props}
  />
))
MenubarTrigger.displayName = MenubarPrimitive.Trigger.displayName

const MenubarSubTrigger = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.SubTrigger>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.SubTrigger> & {
    inset?: boolean
  }
>(({ className, inset, children, ...props }, ref) => (
  <MenubarPrimitive.SubTrigger
    ref={ref}
    className={cn(
      "flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[state=open]:bg-accent data-[state=open]:text-accent-foreground",
      inset && "pl-8",
      className
    )}
    {...props}
  >
    {children}
    <ChevronRight className="ml-auto h-4 w-4" />
  </MenubarPrimitive.SubTrigger>
))
MenubarSubTrigger.displayName = MenubarPrimitive.SubTrigger.displayName

const MenubarSubContent = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.SubContent>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.SubContent>
>(({ className, ...props }, ref) => (
  <MenubarPrimitive.SubContent
    ref={ref}
    className={cn(
      "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))
MenubarSubContent.displayName = MenubarPrimitive.SubContent.displayName

const MenubarContent = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Content>
>(
  (
    { className, align = "start", alignOffset = -4, sideOffset = 8, ...props },
    ref
  ) => (
    <MenubarPrimitive.Portal>
      <MenubarPrimitive.Content
        ref={ref}
        align={align}
        alignOffset={alignOffset}
        sideOffset={sideOffset}
        className={cn(
          "z-50 min-w-[12rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
          className
        )}
        {...props}
      />
    </MenubarPrimitive.Portal>
  )
)
MenubarContent.displayName = MenubarPrimitive.Content.displayName

const MenubarItem = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Item> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <MenubarPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
MenubarItem.displayName = MenubarPrimitive.Item.displayName

const MenubarCheckboxItem = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.CheckboxItem>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.CheckboxItem>
>(({ className, children, checked, ...props }, ref) => (
  <MenubarPrimitive.CheckboxItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    checked={checked}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <MenubarPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </MenubarPrimitive.ItemIndicator>
    </span>
    {children}
  </MenubarPrimitive.CheckboxItem>
))
MenubarCheckboxItem.displayName = MenubarPrimitive.CheckboxItem.displayName

const MenubarRadioItem = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.RadioItem>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.RadioItem>
>(({ className, children, ...props }, ref) => (
  <MenubarPrimitive.RadioItem
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <MenubarPrimitive.ItemIndicator>
        <Circle className="h-2 w-2 fill-current" />
      </MenubarPrimitive.ItemIndicator>
    </span>
    {children}
  </MenubarPrimitive.RadioItem>
))
MenubarRadioItem.displayName = MenubarPrimitive.RadioItem.displayName

const MenubarLabel = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Label> & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  <MenubarPrimitive.Label
    ref={ref}
    className={cn(
      "px-2 py-1.5 text-sm font-semibold",
      inset && "pl-8",
      className
    )}
    {...props}
  />
))
MenubarLabel.displayName = MenubarPrimitive.Label.displayName

const MenubarSeparator = React.forwardRef<
  React.ElementRef<typeof MenubarPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof MenubarPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <MenubarPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
))
MenubarSeparator.displayName = MenubarPrimitive.Separator.displayName

const MenubarShortcut = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) => {
  return (
    <span
      className={cn(
        "ml-auto text-xs tracking-widest text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}
MenubarShortcut.displayname = "MenubarShortcut"

export {
  Menubar,
  MenubarMenu,
  MenubarTrigger,
  MenubarContent,
  MenubarItem,
  MenubarSeparator,
  MenubarLabel,
  MenubarCheckboxItem,
  MenubarRadioGroup,
  MenubarRadioItem,
  MenubarPortal,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarGroup,
  MenubarSub,
  MenubarShortcut,
}

```

# client\src\components\ui\navigation-menu.tsx

```tsx
import * as React from "react"
import * as NavigationMenuPrimitive from "@radix-ui/react-navigation-menu"
import { cva } from "class-variance-authority"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

const NavigationMenu = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.Root>
>(({ className, children, ...props }, ref) => (
  <NavigationMenuPrimitive.Root
    ref={ref}
    className={cn(
      "relative z-10 flex max-w-max flex-1 items-center justify-center",
      className
    )}
    {...props}
  >
    {children}
    <NavigationMenuViewport />
  </NavigationMenuPrimitive.Root>
))
NavigationMenu.displayName = NavigationMenuPrimitive.Root.displayName

const NavigationMenuList = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.List>
>(({ className, ...props }, ref) => (
  <NavigationMenuPrimitive.List
    ref={ref}
    className={cn(
      "group flex flex-1 list-none items-center justify-center space-x-1",
      className
    )}
    {...props}
  />
))
NavigationMenuList.displayName = NavigationMenuPrimitive.List.displayName

const NavigationMenuItem = NavigationMenuPrimitive.Item

const navigationMenuTriggerStyle = cva(
  "group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50"
)

const NavigationMenuTrigger = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <NavigationMenuPrimitive.Trigger
    ref={ref}
    className={cn(navigationMenuTriggerStyle(), "group", className)}
    {...props}
  >
    {children}{" "}
    <ChevronDown
      className="relative top-[1px] ml-1 h-3 w-3 transition duration-200 group-data-[state=open]:rotate-180"
      aria-hidden="true"
    />
  </NavigationMenuPrimitive.Trigger>
))
NavigationMenuTrigger.displayName = NavigationMenuPrimitive.Trigger.displayName

const NavigationMenuContent = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.Content>
>(({ className, ...props }, ref) => (
  <NavigationMenuPrimitive.Content
    ref={ref}
    className={cn(
      "left-0 top-0 w-full data-[motion^=from-]:animate-in data-[motion^=to-]:animate-out data-[motion^=from-]:fade-in data-[motion^=to-]:fade-out data-[motion=from-end]:slide-in-from-right-52 data-[motion=from-start]:slide-in-from-left-52 data-[motion=to-end]:slide-out-to-right-52 data-[motion=to-start]:slide-out-to-left-52 md:absolute md:w-auto ",
      className
    )}
    {...props}
  />
))
NavigationMenuContent.displayName = NavigationMenuPrimitive.Content.displayName

const NavigationMenuLink = NavigationMenuPrimitive.Link

const NavigationMenuViewport = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.Viewport>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.Viewport>
>(({ className, ...props }, ref) => (
  <div className={cn("absolute left-0 top-full flex justify-center")}>
    <NavigationMenuPrimitive.Viewport
      className={cn(
        "origin-top-center relative mt-1.5 h-[var(--radix-navigation-menu-viewport-height)] w-full overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-90 md:w-[var(--radix-navigation-menu-viewport-width)]",
        className
      )}
      ref={ref}
      {...props}
    />
  </div>
))
NavigationMenuViewport.displayName =
  NavigationMenuPrimitive.Viewport.displayName

const NavigationMenuIndicator = React.forwardRef<
  React.ElementRef<typeof NavigationMenuPrimitive.Indicator>,
  React.ComponentPropsWithoutRef<typeof NavigationMenuPrimitive.Indicator>
>(({ className, ...props }, ref) => (
  <NavigationMenuPrimitive.Indicator
    ref={ref}
    className={cn(
      "top-full z-[1] flex h-1.5 items-end justify-center overflow-hidden data-[state=visible]:animate-in data-[state=hidden]:animate-out data-[state=hidden]:fade-out data-[state=visible]:fade-in",
      className
    )}
    {...props}
  >
    <div className="relative top-[60%] h-2 w-2 rotate-45 rounded-tl-sm bg-border shadow-md" />
  </NavigationMenuPrimitive.Indicator>
))
NavigationMenuIndicator.displayName =
  NavigationMenuPrimitive.Indicator.displayName

export {
  navigationMenuTriggerStyle,
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuContent,
  NavigationMenuTrigger,
  NavigationMenuLink,
  NavigationMenuIndicator,
  NavigationMenuViewport,
}

```

# client\src\components\ui\pagination.tsx

```tsx
import * as React from "react"
import { ChevronLeft, ChevronRight, MoreHorizontal } from "lucide-react"

import { cn } from "@/lib/utils"
import { ButtonProps, buttonVariants } from "@/components/ui/button"

const Pagination = ({ className, ...props }: React.ComponentProps<"nav">) => (
  <nav
    role="navigation"
    aria-label="pagination"
    className={cn("mx-auto flex w-full justify-center", className)}
    {...props}
  />
)
Pagination.displayName = "Pagination"

const PaginationContent = React.forwardRef<
  HTMLUListElement,
  React.ComponentProps<"ul">
>(({ className, ...props }, ref) => (
  <ul
    ref={ref}
    className={cn("flex flex-row items-center gap-1", className)}
    {...props}
  />
))
PaginationContent.displayName = "PaginationContent"

const PaginationItem = React.forwardRef<
  HTMLLIElement,
  React.ComponentProps<"li">
>(({ className, ...props }, ref) => (
  <li ref={ref} className={cn("", className)} {...props} />
))
PaginationItem.displayName = "PaginationItem"

type PaginationLinkProps = {
  isActive?: boolean
} & Pick<ButtonProps, "size"> &
  React.ComponentProps<"a">

const PaginationLink = ({
  className,
  isActive,
  size = "icon",
  ...props
}: PaginationLinkProps) => (
  <a
    aria-current={isActive ? "page" : undefined}
    className={cn(
      buttonVariants({
        variant: isActive ? "outline" : "ghost",
        size,
      }),
      className
    )}
    {...props}
  />
)
PaginationLink.displayName = "PaginationLink"

const PaginationPrevious = ({
  className,
  ...props
}: React.ComponentProps<typeof PaginationLink>) => (
  <PaginationLink
    aria-label="Go to previous page"
    size="default"
    className={cn("gap-1 pl-2.5", className)}
    {...props}
  >
    <ChevronLeft className="h-4 w-4" />
    <span>Previous</span>
  </PaginationLink>
)
PaginationPrevious.displayName = "PaginationPrevious"

const PaginationNext = ({
  className,
  ...props
}: React.ComponentProps<typeof PaginationLink>) => (
  <PaginationLink
    aria-label="Go to next page"
    size="default"
    className={cn("gap-1 pr-2.5", className)}
    {...props}
  >
    <span>Next</span>
    <ChevronRight className="h-4 w-4" />
  </PaginationLink>
)
PaginationNext.displayName = "PaginationNext"

const PaginationEllipsis = ({
  className,
  ...props
}: React.ComponentProps<"span">) => (
  <span
    aria-hidden
    className={cn("flex h-9 w-9 items-center justify-center", className)}
    {...props}
  >
    <MoreHorizontal className="h-4 w-4" />
    <span className="sr-only">More pages</span>
  </span>
)
PaginationEllipsis.displayName = "PaginationEllipsis"

export {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
}

```

# client\src\components\ui\popover.tsx

```tsx
import * as React from "react"
import * as PopoverPrimitive from "@radix-ui/react-popover"

import { cn } from "@/lib/utils"

const Popover = PopoverPrimitive.Root

const PopoverTrigger = PopoverPrimitive.Trigger

const PopoverContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>
>(({ className, align = "center", sideOffset = 4, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={cn(
        "z-50 w-72 rounded-md border bg-popover p-4 text-popover-foreground shadow-md outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        className
      )}
      {...props}
    />
  </PopoverPrimitive.Portal>
))
PopoverContent.displayName = PopoverPrimitive.Content.displayName

export { Popover, PopoverTrigger, PopoverContent }

```

# client\src\components\ui\progress.tsx

```tsx
import * as React from "react"
import * as ProgressPrimitive from "@radix-ui/react-progress"

import { cn } from "@/lib/utils"

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, value, ...props }, ref) => (
  <ProgressPrimitive.Root
    ref={ref}
    className={cn(
      "relative h-4 w-full overflow-hidden rounded-full bg-secondary",
      className
    )}
    {...props}
  >
    <ProgressPrimitive.Indicator
      className="h-full w-full flex-1 bg-primary transition-all"
      style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
    />
  </ProgressPrimitive.Root>
))
Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }

```

# client\src\components\ui\radio-group.tsx

```tsx
import * as React from "react"
import * as RadioGroupPrimitive from "@radix-ui/react-radio-group"
import { Circle } from "lucide-react"

import { cn } from "@/lib/utils"

const RadioGroup = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Root>
>(({ className, ...props }, ref) => {
  return (
    <RadioGroupPrimitive.Root
      className={cn("grid gap-2", className)}
      {...props}
      ref={ref}
    />
  )
})
RadioGroup.displayName = RadioGroupPrimitive.Root.displayName

const RadioGroupItem = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Item>
>(({ className, ...props }, ref) => {
  return (
    <RadioGroupPrimitive.Item
      ref={ref}
      className={cn(
        "aspect-square h-4 w-4 rounded-full border border-primary text-primary ring-offset-background focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    >
      <RadioGroupPrimitive.Indicator className="flex items-center justify-center">
        <Circle className="h-2.5 w-2.5 fill-current text-current" />
      </RadioGroupPrimitive.Indicator>
    </RadioGroupPrimitive.Item>
  )
})
RadioGroupItem.displayName = RadioGroupPrimitive.Item.displayName

export { RadioGroup, RadioGroupItem }

```

# client\src\components\ui\RefreshButton.tsx

```tsx
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RefreshButtonProps {
  onClick: () => void;
  className?: string;
}

export function RefreshButton({ onClick, className }: RefreshButtonProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    onClick();
    
    // Reset the animation after a short delay
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleRefresh}
      className={cn(className)}
      disabled={isRefreshing}
    >
      <i className={cn(
        "fas fa-sync-alt",
        isRefreshing && "animate-spin"
      )}></i>
    </Button>
  );
}

```

# client\src\components\ui\resizable.tsx

```tsx
import { GripVertical } from "lucide-react"
import * as ResizablePrimitive from "react-resizable-panels"

import { cn } from "@/lib/utils"

const ResizablePanelGroup = ({
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.PanelGroup>) => (
  <ResizablePrimitive.PanelGroup
    className={cn(
      "flex h-full w-full data-[panel-group-direction=vertical]:flex-col",
      className
    )}
    {...props}
  />
)

const ResizablePanel = ResizablePrimitive.Panel

const ResizableHandle = ({
  withHandle,
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.PanelResizeHandle> & {
  withHandle?: boolean
}) => (
  <ResizablePrimitive.PanelResizeHandle
    className={cn(
      "relative flex w-px items-center justify-center bg-border after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1 data-[panel-group-direction=vertical]:h-px data-[panel-group-direction=vertical]:w-full data-[panel-group-direction=vertical]:after:left-0 data-[panel-group-direction=vertical]:after:h-1 data-[panel-group-direction=vertical]:after:w-full data-[panel-group-direction=vertical]:after:-translate-y-1/2 data-[panel-group-direction=vertical]:after:translate-x-0 [&[data-panel-group-direction=vertical]>div]:rotate-90",
      className
    )}
    {...props}
  >
    {withHandle && (
      <div className="z-10 flex h-4 w-3 items-center justify-center rounded-sm border bg-border">
        <GripVertical className="h-2.5 w-2.5" />
      </div>
    )}
  </ResizablePrimitive.PanelResizeHandle>
)

export { ResizablePanelGroup, ResizablePanel, ResizableHandle }

```

# client\src\components\ui\scroll-area.tsx

```tsx
import * as React from "react"
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"

import { cn } from "@/lib/utils"

const ScrollArea = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>
>(({ className, children, ...props }, ref) => (
  <ScrollAreaPrimitive.Root
    ref={ref}
    className={cn("relative overflow-hidden", className)}
    {...props}
  >
    <ScrollAreaPrimitive.Viewport className="h-full w-full rounded-[inherit]">
      {children}
    </ScrollAreaPrimitive.Viewport>
    <ScrollBar />
    <ScrollAreaPrimitive.Corner />
  </ScrollAreaPrimitive.Root>
))
ScrollArea.displayName = ScrollAreaPrimitive.Root.displayName

const ScrollBar = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>
>(({ className, orientation = "vertical", ...props }, ref) => (
  <ScrollAreaPrimitive.ScrollAreaScrollbar
    ref={ref}
    orientation={orientation}
    className={cn(
      "flex touch-none select-none transition-colors",
      orientation === "vertical" &&
        "h-full w-2.5 border-l border-l-transparent p-[1px]",
      orientation === "horizontal" &&
        "h-2.5 flex-col border-t border-t-transparent p-[1px]",
      className
    )}
    {...props}
  >
    <ScrollAreaPrimitive.ScrollAreaThumb className="relative flex-1 rounded-full bg-border" />
  </ScrollAreaPrimitive.ScrollAreaScrollbar>
))
ScrollBar.displayName = ScrollAreaPrimitive.ScrollAreaScrollbar.displayName

export { ScrollArea, ScrollBar }

```

# client\src\components\ui\select.tsx

```tsx
import * as React from "react"
import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown, ChevronUp } from "lucide-react"

import { cn } from "@/lib/utils"

const Select = SelectPrimitive.Root

const SelectGroup = SelectPrimitive.Group

const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronUp className="h-4 w-4" />
  </SelectPrimitive.ScrollUpButton>
))
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronDown className="h-4 w-4" />
  </SelectPrimitive.ScrollDownButton>
))
SelectScrollDownButton.displayName =
  SelectPrimitive.ScrollDownButton.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("py-1.5 pl-8 pr-2 text-sm font-semibold", className)}
    {...props}
  />
))
SelectLabel.displayName = SelectPrimitive.Label.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>

    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
))
SelectSeparator.displayName = SelectPrimitive.Separator.displayName

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
}

```

# client\src\components\ui\separator.tsx

```tsx
import * as React from "react"
import * as SeparatorPrimitive from "@radix-ui/react-separator"

import { cn } from "@/lib/utils"

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>
>(
  (
    { className, orientation = "horizontal", decorative = true, ...props },
    ref
  ) => (
    <SeparatorPrimitive.Root
      ref={ref}
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]",
        className
      )}
      {...props}
    />
  )
)
Separator.displayName = SeparatorPrimitive.Root.displayName

export { Separator }

```

# client\src\components\ui\sheet.tsx

```tsx
import * as React from "react"
import * as SheetPrimitive from "@radix-ui/react-dialog"
import { cva, type VariantProps } from "class-variance-authority"
import { X } from "lucide-react"

import { cn } from "@/lib/utils"

const Sheet = SheetPrimitive.Root

const SheetTrigger = SheetPrimitive.Trigger

const SheetClose = SheetPrimitive.Close

const SheetPortal = SheetPrimitive.Portal

const SheetOverlay = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Overlay
    className={cn(
      "fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
    ref={ref}
  />
))
SheetOverlay.displayName = SheetPrimitive.Overlay.displayName

const sheetVariants = cva(
  "fixed z-50 gap-4 bg-background p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500",
  {
    variants: {
      side: {
        top: "inset-x-0 top-0 border-b data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top",
        bottom:
          "inset-x-0 bottom-0 border-t data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom",
        left: "inset-y-0 left-0 h-full w-3/4 border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm",
        right:
          "inset-y-0 right-0 h-full w-3/4  border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-sm",
      },
    },
    defaultVariants: {
      side: "right",
    },
  }
)

interface SheetContentProps
  extends React.ComponentPropsWithoutRef<typeof SheetPrimitive.Content>,
    VariantProps<typeof sheetVariants> {}

const SheetContent = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Content>,
  SheetContentProps
>(({ side = "right", className, children, ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <SheetPrimitive.Content
      ref={ref}
      className={cn(sheetVariants({ side }), className)}
      {...props}
    >
      {children}
      <SheetPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-secondary">
        <X className="h-4 w-4" />
        <span className="sr-only">Close</span>
      </SheetPrimitive.Close>
    </SheetPrimitive.Content>
  </SheetPortal>
))
SheetContent.displayName = SheetPrimitive.Content.displayName

const SheetHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col space-y-2 text-center sm:text-left",
      className
    )}
    {...props}
  />
)
SheetHeader.displayName = "SheetHeader"

const SheetFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
      className
    )}
    {...props}
  />
)
SheetFooter.displayName = "SheetFooter"

const SheetTitle = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Title>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Title
    ref={ref}
    className={cn("text-lg font-semibold text-foreground", className)}
    {...props}
  />
))
SheetTitle.displayName = SheetPrimitive.Title.displayName

const SheetDescription = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Description>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Description
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
SheetDescription.displayName = SheetPrimitive.Description.displayName

export {
  Sheet,
  SheetPortal,
  SheetOverlay,
  SheetTrigger,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetFooter,
  SheetTitle,
  SheetDescription,
}

```

# client\src\components\ui\sidebar.tsx

```tsx
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { VariantProps, cva } from "class-variance-authority"
import { PanelLeft } from "lucide-react"

import { useIsMobile } from "@/hooks/use-mobile"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

const SIDEBAR_COOKIE_NAME = "sidebar:state"
const SIDEBAR_COOKIE_MAX_AGE = 60 * 60 * 24 * 7
const SIDEBAR_WIDTH = "16rem"
const SIDEBAR_WIDTH_MOBILE = "18rem"
const SIDEBAR_WIDTH_ICON = "3rem"
const SIDEBAR_KEYBOARD_SHORTCUT = "b"

type SidebarContext = {
  state: "expanded" | "collapsed"
  open: boolean
  setOpen: (open: boolean) => void
  openMobile: boolean
  setOpenMobile: (open: boolean) => void
  isMobile: boolean
  toggleSidebar: () => void
}

const SidebarContext = React.createContext<SidebarContext | null>(null)

function useSidebar() {
  const context = React.useContext(SidebarContext)
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider.")
  }

  return context
}

const SidebarProvider = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    defaultOpen?: boolean
    open?: boolean
    onOpenChange?: (open: boolean) => void
  }
>(
  (
    {
      defaultOpen = true,
      open: openProp,
      onOpenChange: setOpenProp,
      className,
      style,
      children,
      ...props
    },
    ref
  ) => {
    const isMobile = useIsMobile()
    const [openMobile, setOpenMobile] = React.useState(false)

    // This is the internal state of the sidebar.
    // We use openProp and setOpenProp for control from outside the component.
    const [_open, _setOpen] = React.useState(defaultOpen)
    const open = openProp ?? _open
    const setOpen = React.useCallback(
      (value: boolean | ((value: boolean) => boolean)) => {
        if (setOpenProp) {
          return setOpenProp?.(
            typeof value === "function" ? value(open) : value
          )
        }

        _setOpen(value)

        // This sets the cookie to keep the sidebar state.
        document.cookie = `${SIDEBAR_COOKIE_NAME}=${open}; path=/; max-age=${SIDEBAR_COOKIE_MAX_AGE}`
      },
      [setOpenProp, open]
    )

    // Helper to toggle the sidebar.
    const toggleSidebar = React.useCallback(() => {
      return isMobile
        ? setOpenMobile((open) => !open)
        : setOpen((open) => !open)
    }, [isMobile, setOpen, setOpenMobile])

    // Adds a keyboard shortcut to toggle the sidebar.
    React.useEffect(() => {
      const handleKeyDown = (event: KeyboardEvent) => {
        if (
          event.key === SIDEBAR_KEYBOARD_SHORTCUT &&
          (event.metaKey || event.ctrlKey)
        ) {
          event.preventDefault()
          toggleSidebar()
        }
      }

      window.addEventListener("keydown", handleKeyDown)
      return () => window.removeEventListener("keydown", handleKeyDown)
    }, [toggleSidebar])

    // We add a state so that we can do data-state="expanded" or "collapsed".
    // This makes it easier to style the sidebar with Tailwind classes.
    const state = open ? "expanded" : "collapsed"

    const contextValue = React.useMemo<SidebarContext>(
      () => ({
        state,
        open,
        setOpen,
        isMobile,
        openMobile,
        setOpenMobile,
        toggleSidebar,
      }),
      [state, open, setOpen, isMobile, openMobile, setOpenMobile, toggleSidebar]
    )

    return (
      <SidebarContext.Provider value={contextValue}>
        <TooltipProvider delayDuration={0}>
          <div
            style={
              {
                "--sidebar-width": SIDEBAR_WIDTH,
                "--sidebar-width-icon": SIDEBAR_WIDTH_ICON,
                ...style,
              } as React.CSSProperties
            }
            className={cn(
              "group/sidebar-wrapper flex min-h-svh w-full text-sidebar-foreground has-[[data-variant=inset]]:bg-sidebar",
              className
            )}
            ref={ref}
            {...props}
          >
            {children}
          </div>
        </TooltipProvider>
      </SidebarContext.Provider>
    )
  }
)
SidebarProvider.displayName = "SidebarProvider"

const Sidebar = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    side?: "left" | "right"
    variant?: "sidebar" | "floating" | "inset"
    collapsible?: "offcanvas" | "icon" | "none"
  }
>(
  (
    {
      side = "left",
      variant = "sidebar",
      collapsible = "offcanvas",
      className,
      children,
      ...props
    },
    ref
  ) => {
    const { isMobile, state, openMobile, setOpenMobile } = useSidebar()

    if (collapsible === "none") {
      return (
        <div
          className={cn(
            "flex h-full w-[--sidebar-width] flex-col bg-sidebar text-sidebar-foreground",
            className
          )}
          ref={ref}
          {...props}
        >
          {children}
        </div>
      )
    }

    if (isMobile) {
      return (
        <Sheet open={openMobile} onOpenChange={setOpenMobile} {...props}>
          <SheetContent
            data-sidebar="sidebar"
            data-mobile="true"
            className="w-[--sidebar-width] bg-sidebar p-0 text-sidebar-foreground [&>button]:hidden"
            style={
              {
                "--sidebar-width": SIDEBAR_WIDTH_MOBILE,
              } as React.CSSProperties
            }
            side={side}
          >
            <div className="flex h-full w-full flex-col">{children}</div>
          </SheetContent>
        </Sheet>
      )
    }

    return (
      <div
        ref={ref}
        className="group peer hidden md:block"
        data-state={state}
        data-collapsible={state === "collapsed" ? collapsible : ""}
        data-variant={variant}
        data-side={side}
      >
        {/* This is what handles the sidebar gap on desktop */}
        <div
          className={cn(
            "duration-200 relative h-svh w-[--sidebar-width] bg-transparent transition-[width] ease-linear",
            "group-data-[collapsible=offcanvas]:w-0",
            "group-data-[side=right]:rotate-180",
            variant === "floating" || variant === "inset"
              ? "group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)_+_theme(spacing.4))]"
              : "group-data-[collapsible=icon]:w-[--sidebar-width-icon]"
          )}
        />
        <div
          className={cn(
            "duration-200 fixed inset-y-0 z-10 hidden h-svh w-[--sidebar-width] transition-[left,right,width] ease-linear md:flex",
            side === "left"
              ? "left-0 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)]"
              : "right-0 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]",
            // Adjust the padding for floating and inset variants.
            variant === "floating" || variant === "inset"
              ? "p-2 group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)_+_theme(spacing.4)_+2px)]"
              : "group-data-[collapsible=icon]:w-[--sidebar-width-icon] group-data-[side=left]:border-r group-data-[side=right]:border-l",
            className
          )}
          {...props}
        >
          <div
            data-sidebar="sidebar"
            className="flex h-full w-full flex-col bg-sidebar group-data-[variant=floating]:rounded-lg group-data-[variant=floating]:border group-data-[variant=floating]:border-sidebar-border group-data-[variant=floating]:shadow"
          >
            {children}
          </div>
        </div>
      </div>
    )
  }
)
Sidebar.displayName = "Sidebar"

const SidebarTrigger = React.forwardRef<
  React.ElementRef<typeof Button>,
  React.ComponentProps<typeof Button>
>(({ className, onClick, ...props }, ref) => {
  const { toggleSidebar } = useSidebar()

  return (
    <Button
      ref={ref}
      data-sidebar="trigger"
      variant="ghost"
      size="icon"
      className={cn("h-7 w-7", className)}
      onClick={(event) => {
        onClick?.(event)
        toggleSidebar()
      }}
      {...props}
    >
      <PanelLeft />
      <span className="sr-only">Toggle Sidebar</span>
    </Button>
  )
})
SidebarTrigger.displayName = "SidebarTrigger"

const SidebarRail = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button">
>(({ className, ...props }, ref) => {
  const { toggleSidebar } = useSidebar()

  return (
    <button
      ref={ref}
      data-sidebar="rail"
      aria-label="Toggle Sidebar"
      tabIndex={-1}
      onClick={toggleSidebar}
      title="Toggle Sidebar"
      className={cn(
        "absolute inset-y-0 z-20 hidden w-4 -translate-x-1/2 transition-all ease-linear after:absolute after:inset-y-0 after:left-1/2 after:w-[2px] hover:after:bg-sidebar-border group-data-[side=left]:-right-4 group-data-[side=right]:left-0 sm:flex",
        "[[data-side=left]_&]:cursor-w-resize [[data-side=right]_&]:cursor-e-resize",
        "[[data-side=left][data-state=collapsed]_&]:cursor-e-resize [[data-side=right][data-state=collapsed]_&]:cursor-w-resize",
        "group-data-[collapsible=offcanvas]:translate-x-0 group-data-[collapsible=offcanvas]:after:left-full group-data-[collapsible=offcanvas]:hover:bg-sidebar",
        "[[data-side=left][data-collapsible=offcanvas]_&]:-right-2",
        "[[data-side=right][data-collapsible=offcanvas]_&]:-left-2",
        className
      )}
      {...props}
    />
  )
})
SidebarRail.displayName = "SidebarRail"

const SidebarInset = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"main">
>(({ className, ...props }, ref) => {
  return (
    <main
      ref={ref}
      className={cn(
        "relative flex min-h-svh flex-1 flex-col bg-background",
        "peer-data-[variant=inset]:min-h-[calc(100svh-theme(spacing.4))] md:peer-data-[variant=inset]:m-2 md:peer-data-[state=collapsed]:peer-data-[variant=inset]:ml-2 md:peer-data-[variant=inset]:ml-0 md:peer-data-[variant=inset]:rounded-xl md:peer-data-[variant=inset]:shadow",
        className
      )}
      {...props}
    />
  )
})
SidebarInset.displayName = "SidebarInset"

const SidebarInput = React.forwardRef<
  React.ElementRef<typeof Input>,
  React.ComponentProps<typeof Input>
>(({ className, ...props }, ref) => {
  return (
    <Input
      ref={ref}
      data-sidebar="input"
      className={cn(
        "h-8 w-full bg-background shadow-none focus-visible:ring-2 focus-visible:ring-sidebar-ring",
        className
      )}
      {...props}
    />
  )
})
SidebarInput.displayName = "SidebarInput"

const SidebarHeader = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      data-sidebar="header"
      className={cn("flex flex-col gap-2 p-2", className)}
      {...props}
    />
  )
})
SidebarHeader.displayName = "SidebarHeader"

const SidebarFooter = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      data-sidebar="footer"
      className={cn("flex flex-col gap-2 p-2", className)}
      {...props}
    />
  )
})
SidebarFooter.displayName = "SidebarFooter"

const SidebarSeparator = React.forwardRef<
  React.ElementRef<typeof Separator>,
  React.ComponentProps<typeof Separator>
>(({ className, ...props }, ref) => {
  return (
    <Separator
      ref={ref}
      data-sidebar="separator"
      className={cn("mx-2 w-auto bg-sidebar-border", className)}
      {...props}
    />
  )
})
SidebarSeparator.displayName = "SidebarSeparator"

const SidebarContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      data-sidebar="content"
      className={cn(
        "flex min-h-0 flex-1 flex-col gap-2 overflow-auto group-data-[collapsible=icon]:overflow-hidden",
        className
      )}
      {...props}
    />
  )
})
SidebarContent.displayName = "SidebarContent"

const SidebarGroup = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      data-sidebar="group"
      className={cn("relative flex w-full min-w-0 flex-col p-2", className)}
      {...props}
    />
  )
})
SidebarGroup.displayName = "SidebarGroup"

const SidebarGroupLabel = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & { asChild?: boolean }
>(({ className, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "div"

  return (
    <Comp
      ref={ref}
      data-sidebar="group-label"
      className={cn(
        "duration-200 flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium text-sidebar-foreground/70 outline-none ring-sidebar-ring transition-[margin,opa] ease-linear focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0",
        "group-data-[collapsible=icon]:-mt-8 group-data-[collapsible=icon]:opacity-0",
        className
      )}
      {...props}
    />
  )
})
SidebarGroupLabel.displayName = "SidebarGroupLabel"

const SidebarGroupAction = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button"> & { asChild?: boolean }
>(({ className, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      ref={ref}
      data-sidebar="group-action"
      className={cn(
        "absolute right-3 top-3.5 flex aspect-square w-5 items-center justify-center rounded-md p-0 text-sidebar-foreground outline-none ring-sidebar-ring transition-transform hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0",
        // Increases the hit area of the button on mobile.
        "after:absolute after:-inset-2 after:md:hidden",
        "group-data-[collapsible=icon]:hidden",
        className
      )}
      {...props}
    />
  )
})
SidebarGroupAction.displayName = "SidebarGroupAction"

const SidebarGroupContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-sidebar="group-content"
    className={cn("w-full text-sm", className)}
    {...props}
  />
))
SidebarGroupContent.displayName = "SidebarGroupContent"

const SidebarMenu = React.forwardRef<
  HTMLUListElement,
  React.ComponentProps<"ul">
>(({ className, ...props }, ref) => (
  <ul
    ref={ref}
    data-sidebar="menu"
    className={cn("flex w-full min-w-0 flex-col gap-1", className)}
    {...props}
  />
))
SidebarMenu.displayName = "SidebarMenu"

const SidebarMenuItem = React.forwardRef<
  HTMLLIElement,
  React.ComponentProps<"li">
>(({ className, ...props }, ref) => (
  <li
    ref={ref}
    data-sidebar="menu-item"
    className={cn("group/menu-item relative", className)}
    {...props}
  />
))
SidebarMenuItem.displayName = "SidebarMenuItem"

const sidebarMenuButtonVariants = cva(
  "peer/menu-button flex w-full items-center gap-2 overflow-hidden rounded-md p-2 text-left text-sm outline-none ring-sidebar-ring transition-[width,height,padding] hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 active:bg-sidebar-accent active:text-sidebar-accent-foreground disabled:pointer-events-none disabled:opacity-50 group-has-[[data-sidebar=menu-action]]/menu-item:pr-8 aria-disabled:pointer-events-none aria-disabled:opacity-50 data-[active=true]:bg-sidebar-accent data-[active=true]:font-medium data-[active=true]:text-sidebar-accent-foreground data-[state=open]:hover:bg-sidebar-accent data-[state=open]:hover:text-sidebar-accent-foreground group-data-[collapsible=icon]:!size-8 group-data-[collapsible=icon]:!p-2 [&>span:last-child]:truncate [&>svg]:size-4 [&>svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
        outline:
          "bg-background shadow-[0_0_0_1px_hsl(var(--sidebar-border))] hover:bg-sidebar-accent hover:text-sidebar-accent-foreground hover:shadow-[0_0_0_1px_hsl(var(--sidebar-accent))]",
      },
      size: {
        default: "h-8 text-sm",
        sm: "h-7 text-xs",
        lg: "h-12 text-sm group-data-[collapsible=icon]:!p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

const SidebarMenuButton = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button"> & {
    asChild?: boolean
    isActive?: boolean
    tooltip?: string | React.ComponentProps<typeof TooltipContent>
  } & VariantProps<typeof sidebarMenuButtonVariants>
>(
  (
    {
      asChild = false,
      isActive = false,
      variant = "default",
      size = "default",
      tooltip,
      className,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : "button"
    const { isMobile, state } = useSidebar()

    const button = (
      <Comp
        ref={ref}
        data-sidebar="menu-button"
        data-size={size}
        data-active={isActive}
        className={cn(sidebarMenuButtonVariants({ variant, size }), className)}
        {...props}
      />
    )

    if (!tooltip) {
      return button
    }

    if (typeof tooltip === "string") {
      tooltip = {
        children: tooltip,
      }
    }

    return (
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent
          side="right"
          align="center"
          hidden={state !== "collapsed" || isMobile}
          {...tooltip}
        />
      </Tooltip>
    )
  }
)
SidebarMenuButton.displayName = "SidebarMenuButton"

const SidebarMenuAction = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button"> & {
    asChild?: boolean
    showOnHover?: boolean
  }
>(({ className, asChild = false, showOnHover = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      ref={ref}
      data-sidebar="menu-action"
      className={cn(
        "absolute right-1 top-1.5 flex aspect-square w-5 items-center justify-center rounded-md p-0 text-sidebar-foreground outline-none ring-sidebar-ring transition-transform hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 peer-hover/menu-button:text-sidebar-accent-foreground [&>svg]:size-4 [&>svg]:shrink-0",
        // Increases the hit area of the button on mobile.
        "after:absolute after:-inset-2 after:md:hidden",
        "peer-data-[size=sm]/menu-button:top-1",
        "peer-data-[size=default]/menu-button:top-1.5",
        "peer-data-[size=lg]/menu-button:top-2.5",
        "group-data-[collapsible=icon]:hidden",
        showOnHover &&
          "group-focus-within/menu-item:opacity-100 group-hover/menu-item:opacity-100 data-[state=open]:opacity-100 peer-data-[active=true]/menu-button:text-sidebar-accent-foreground md:opacity-0",
        className
      )}
      {...props}
    />
  )
})
SidebarMenuAction.displayName = "SidebarMenuAction"

const SidebarMenuBadge = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-sidebar="menu-badge"
    className={cn(
      "absolute right-1 flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-xs font-medium tabular-nums text-sidebar-foreground select-none pointer-events-none",
      "peer-hover/menu-button:text-sidebar-accent-foreground peer-data-[active=true]/menu-button:text-sidebar-accent-foreground",
      "peer-data-[size=sm]/menu-button:top-1",
      "peer-data-[size=default]/menu-button:top-1.5",
      "peer-data-[size=lg]/menu-button:top-2.5",
      "group-data-[collapsible=icon]:hidden",
      className
    )}
    {...props}
  />
))
SidebarMenuBadge.displayName = "SidebarMenuBadge"

const SidebarMenuSkeleton = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    showIcon?: boolean
  }
>(({ className, showIcon = false, ...props }, ref) => {
  // Random width between 50 to 90%.
  const width = React.useMemo(() => {
    return `${Math.floor(Math.random() * 40) + 50}%`
  }, [])

  return (
    <div
      ref={ref}
      data-sidebar="menu-skeleton"
      className={cn("rounded-md h-8 flex gap-2 px-2 items-center", className)}
      {...props}
    >
      {showIcon && (
        <Skeleton
          className="size-4 rounded-md"
          data-sidebar="menu-skeleton-icon"
        />
      )}
      <Skeleton
        className="h-4 flex-1 max-w-[--skeleton-width]"
        data-sidebar="menu-skeleton-text"
        style={
          {
            "--skeleton-width": width,
          } as React.CSSProperties
        }
      />
    </div>
  )
})
SidebarMenuSkeleton.displayName = "SidebarMenuSkeleton"

const SidebarMenuSub = React.forwardRef<
  HTMLUListElement,
  React.ComponentProps<"ul">
>(({ className, ...props }, ref) => (
  <ul
    ref={ref}
    data-sidebar="menu-sub"
    className={cn(
      "mx-3.5 flex min-w-0 translate-x-px flex-col gap-1 border-l border-sidebar-border px-2.5 py-0.5",
      "group-data-[collapsible=icon]:hidden",
      className
    )}
    {...props}
  />
))
SidebarMenuSub.displayName = "SidebarMenuSub"

const SidebarMenuSubItem = React.forwardRef<
  HTMLLIElement,
  React.ComponentProps<"li">
>(({ ...props }, ref) => <li ref={ref} {...props} />)
SidebarMenuSubItem.displayName = "SidebarMenuSubItem"

const SidebarMenuSubButton = React.forwardRef<
  HTMLAnchorElement,
  React.ComponentProps<"a"> & {
    asChild?: boolean
    size?: "sm" | "md"
    isActive?: boolean
  }
>(({ asChild = false, size = "md", isActive, className, ...props }, ref) => {
  const Comp = asChild ? Slot : "a"

  return (
    <Comp
      ref={ref}
      data-sidebar="menu-sub-button"
      data-size={size}
      data-active={isActive}
      className={cn(
        "flex h-7 min-w-0 -translate-x-px items-center gap-2 overflow-hidden rounded-md px-2 text-sidebar-foreground outline-none ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 active:bg-sidebar-accent active:text-sidebar-accent-foreground disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&>span:last-child]:truncate [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:text-sidebar-accent-foreground",
        "data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground",
        size === "sm" && "text-xs",
        size === "md" && "text-sm",
        "group-data-[collapsible=icon]:hidden",
        className
      )}
      {...props}
    />
  )
})
SidebarMenuSubButton.displayName = "SidebarMenuSubButton"

export {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarInset,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
}

```

# client\src\components\ui\skeleton.tsx

```tsx
import React from "react";
import { cn } from "@/lib/utils"

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }

```

# client\src\components\ui\slider.tsx

```tsx
import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
      <SliderPrimitive.Range className="absolute h-full bg-primary" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>
))
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }

```

# client\src\components\ui\StatusIndicator.tsx

```tsx
import React from 'react';
import { cn } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'error' | 'checking';
  pulsing?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

export function StatusIndicator({ 
  status, 
  pulsing = true, 
  size = 'sm',
  label
}: StatusIndicatorProps) {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  const statusClasses = {
    healthy: 'bg-secondary',
    warning: 'bg-yellow-400',
    error: 'bg-destructive',
    checking: 'bg-gray-400'
  };

  const statusTextClasses = {
    healthy: 'text-secondary',
    warning: 'text-yellow-400',
    error: 'text-destructive',
    checking: 'text-gray-400'
  };

  return (
    <div className="flex items-center">
      <div 
        className={cn(
          "rounded-full mr-1",
          sizeClasses[size],
          statusClasses[status],
          pulsing && "pulse"
        )}
      ></div>
      {label && (
        <span className={cn("text-xs", statusTextClasses[status])}>
          {label}
        </span>
      )}
    </div>
  );
}

```

# client\src\components\ui\switch.tsx

```tsx
import * as React from "react"
import * as SwitchPrimitives from "@radix-ui/react-switch"

import { cn } from "@/lib/utils"

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(
      "peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-input",
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0"
      )}
    />
  </SwitchPrimitives.Root>
))
Switch.displayName = SwitchPrimitives.Root.displayName

export { Switch }

```

# client\src\components\ui\table.tsx

```tsx
import * as React from "react"

import { cn } from "@/lib/utils"

const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <div className="relative w-full overflow-auto">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
))
Table.displayName = "Table"

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("[&_tr]:border-b", className)} {...props} />
))
TableHeader.displayName = "TableHeader"

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0", className)}
    {...props}
  />
))
TableBody.displayName = "TableBody"

const TableFooter = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tfoot
    ref={ref}
    className={cn(
      "border-t bg-muted/50 font-medium [&>tr]:last:border-b-0",
      className
    )}
    {...props}
  />
))
TableFooter.displayName = "TableFooter"

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted",
      className
    )}
    {...props}
  />
))
TableRow.displayName = "TableRow"

const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0",
      className
    )}
    {...props}
  />
))
TableHead.displayName = "TableHead"

const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn("p-4 align-middle [&:has([role=checkbox])]:pr-0", className)}
    {...props}
  />
))
TableCell.displayName = "TableCell"

const TableCaption = React.forwardRef<
  HTMLTableCaptionElement,
  React.HTMLAttributes<HTMLTableCaptionElement>
>(({ className, ...props }, ref) => (
  <caption
    ref={ref}
    className={cn("mt-4 text-sm text-muted-foreground", className)}
    {...props}
  />
))
TableCaption.displayName = "TableCaption"

export {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption,
}

```

# client\src\components\ui\tabs.tsx

```tsx
import * as React from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"

import { cn } from "@/lib/utils"

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
      className
    )}
    {...props}
  />
))
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
      className
    )}
    {...props}
  />
))
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
      className
    )}
    {...props}
  />
))
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }

```

# client\src\components\ui\textarea.tsx

```tsx
import * as React from "react"

import { cn } from "@/lib/utils"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

export { Textarea }

```

# client\src\components\ui\toast.tsx

```tsx
import * as React from "react"
import * as ToastPrimitives from "@radix-ui/react-toast"
import { cva, type VariantProps } from "class-variance-authority"
import { X } from "lucide-react"

import { cn } from "@/lib/utils"

const ToastProvider = ToastPrimitives.Provider

const ToastViewport = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Viewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    ref={ref}
    className={cn(
      "fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]",
      className
    )}
    {...props}
  />
))
ToastViewport.displayName = ToastPrimitives.Viewport.displayName

const toastVariants = cva(
  "group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full",
  {
    variants: {
      variant: {
        default: "border bg-background text-foreground",
        destructive:
          "destructive group border-destructive bg-destructive text-destructive-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Root> &
    VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => {
  return (
    <ToastPrimitives.Root
      ref={ref}
      className={cn(toastVariants({ variant }), className)}
      {...props}
    />
  )
})
Toast.displayName = ToastPrimitives.Root.displayName

const ToastAction = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Action>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Action>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Action
    ref={ref}
    className={cn(
      "inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium ring-offset-background transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 group-[.destructive]:border-muted/40 group-[.destructive]:hover:border-destructive/30 group-[.destructive]:hover:bg-destructive group-[.destructive]:hover:text-destructive-foreground group-[.destructive]:focus:ring-destructive",
      className
    )}
    {...props}
  />
))
ToastAction.displayName = ToastPrimitives.Action.displayName

const ToastClose = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Close>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    ref={ref}
    className={cn(
      "absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50 group-[.destructive]:focus:ring-red-400 group-[.destructive]:focus:ring-offset-red-600",
      className
    )}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.Close>
))
ToastClose.displayName = ToastPrimitives.Close.displayName

const ToastTitle = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Title>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    ref={ref}
    className={cn("text-sm font-semibold", className)}
    {...props}
  />
))
ToastTitle.displayName = ToastPrimitives.Title.displayName

const ToastDescription = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Description>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    ref={ref}
    className={cn("text-sm opacity-90", className)}
    {...props}
  />
))
ToastDescription.displayName = ToastPrimitives.Description.displayName

type ToastProps = React.ComponentPropsWithoutRef<typeof Toast>

type ToastActionElement = React.ReactElement<typeof ToastAction>

export {
  type ToastProps,
  type ToastActionElement,
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
}

```

# client\src\components\ui\toaster.tsx

```tsx
import React from "react";
import { useToast } from "@/hooks/use-toast"
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast"

export function Toaster() {
  const { toasts } = useToast()

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          <Toast key={id} {...props}>
            <div className="grid gap-1">
              {title && <ToastTitle>{title}</ToastTitle>}
              {description && (
                <ToastDescription>{description}</ToastDescription>
              )}
            </div>
            {action}
            <ToastClose />
          </Toast>
        )
      })}
      <ToastViewport />
    </ToastProvider>
  )
}

```

# client\src\components\ui\toggle-group.tsx

```tsx
import * as React from "react"
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group"
import { type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"
import { toggleVariants } from "@/components/ui/toggle"

const ToggleGroupContext = React.createContext<
  VariantProps<typeof toggleVariants>
>({
  size: "default",
  variant: "default",
})

const ToggleGroup = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Root> &
    VariantProps<typeof toggleVariants>
>(({ className, variant, size, children, ...props }, ref) => (
  <ToggleGroupPrimitive.Root
    ref={ref}
    className={cn("flex items-center justify-center gap-1", className)}
    {...props}
  >
    <ToggleGroupContext.Provider value={{ variant, size }}>
      {children}
    </ToggleGroupContext.Provider>
  </ToggleGroupPrimitive.Root>
))

ToggleGroup.displayName = ToggleGroupPrimitive.Root.displayName

const ToggleGroupItem = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Item> &
    VariantProps<typeof toggleVariants>
>(({ className, children, variant, size, ...props }, ref) => {
  const context = React.useContext(ToggleGroupContext)

  return (
    <ToggleGroupPrimitive.Item
      ref={ref}
      className={cn(
        toggleVariants({
          variant: context.variant || variant,
          size: context.size || size,
        }),
        className
      )}
      {...props}
    >
      {children}
    </ToggleGroupPrimitive.Item>
  )
})

ToggleGroupItem.displayName = ToggleGroupPrimitive.Item.displayName

export { ToggleGroup, ToggleGroupItem }

```

# client\src\components\ui\toggle.tsx

```tsx
import * as React from "react"
import * as TogglePrimitive from "@radix-ui/react-toggle"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const toggleVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors hover:bg-muted hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground",
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline:
          "border border-input bg-transparent hover:bg-accent hover:text-accent-foreground",
      },
      size: {
        default: "h-10 px-3",
        sm: "h-9 px-2.5",
        lg: "h-11 px-5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

const Toggle = React.forwardRef<
  React.ElementRef<typeof TogglePrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof TogglePrimitive.Root> &
    VariantProps<typeof toggleVariants>
>(({ className, variant, size, ...props }, ref) => (
  <TogglePrimitive.Root
    ref={ref}
    className={cn(toggleVariants({ variant, size, className }))}
    {...props}
  />
))

Toggle.displayName = TogglePrimitive.Root.displayName

export { Toggle, toggleVariants }

```

# client\src\components\ui\tooltip.tsx

```tsx
import * as React from "react"
import * as TooltipPrimitive from "@radix-ui/react-tooltip"

import { cn } from "@/lib/utils"

const TooltipProvider = TooltipPrimitive.Provider

const Tooltip = TooltipPrimitive.Root

const TooltipTrigger = TooltipPrimitive.Trigger

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))
TooltipContent.displayName = TooltipPrimitive.Content.displayName

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider }

```

# client\src\contexts\FeaturesContext.tsx

```tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRuntimeConfig } from '../lib/api';

type FeaturesContextType = {
  explainabilityEnabled: boolean;
  isLoading: boolean;
};

const FeaturesContext = createContext<FeaturesContextType>({
  explainabilityEnabled: false,
  isLoading: true,
});

export const FeaturesProvider = ({ children }: { children: ReactNode }) => {
  const { data, isLoading } = useRuntimeConfig('memory-core');
  const [explainabilityEnabled, setExplainabilityEnabled] = useState(false);
  
  useEffect(() => {
    if (data?.config) {
      // Check if the ENABLE_EXPLAINABILITY flag is present in the config
      setExplainabilityEnabled(!!data.config.ENABLE_EXPLAINABILITY);
    }
  }, [data]);
  
  return (
    <FeaturesContext.Provider value={{ explainabilityEnabled, isLoading }}>
      {children}
    </FeaturesContext.Provider>
  );
};

export const useFeatures = () => useContext(FeaturesContext);

```

# client\src\hooks\use-mobile.tsx

```tsx
import * as React from "react"

const MOBILE_BREAKPOINT = 768

export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState<boolean | undefined>(undefined)

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const onChange = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    }
    mql.addEventListener("change", onChange)
    setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    return () => mql.removeEventListener("change", onChange)
  }, [])

  return !!isMobile
}

```

# client\src\hooks\use-toast.ts

```ts
import * as React from "react"

import type {
  ToastActionElement,
  ToastProps,
} from "@/components/ui/toast"

const TOAST_LIMIT = 1
const TOAST_REMOVE_DELAY = 1000000

type ToasterToast = ToastProps & {
  id: string
  title?: React.ReactNode
  description?: React.ReactNode
  action?: ToastActionElement
}

const actionTypes = {
  ADD_TOAST: "ADD_TOAST",
  UPDATE_TOAST: "UPDATE_TOAST",
  DISMISS_TOAST: "DISMISS_TOAST",
  REMOVE_TOAST: "REMOVE_TOAST",
} as const

let count = 0

function genId() {
  count = (count + 1) % Number.MAX_SAFE_INTEGER
  return count.toString()
}

type ActionType = typeof actionTypes

type Action =
  | {
      type: ActionType["ADD_TOAST"]
      toast: ToasterToast
    }
  | {
      type: ActionType["UPDATE_TOAST"]
      toast: Partial<ToasterToast>
    }
  | {
      type: ActionType["DISMISS_TOAST"]
      toastId?: ToasterToast["id"]
    }
  | {
      type: ActionType["REMOVE_TOAST"]
      toastId?: ToasterToast["id"]
    }

interface State {
  toasts: ToasterToast[]
}

const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>()

const addToRemoveQueue = (toastId: string) => {
  if (toastTimeouts.has(toastId)) {
    return
  }

  const timeout = setTimeout(() => {
    toastTimeouts.delete(toastId)
    dispatch({
      type: "REMOVE_TOAST",
      toastId: toastId,
    })
  }, TOAST_REMOVE_DELAY)

  toastTimeouts.set(toastId, timeout)
}

export const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case "ADD_TOAST":
      return {
        ...state,
        toasts: [action.toast, ...state.toasts].slice(0, TOAST_LIMIT),
      }

    case "UPDATE_TOAST":
      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === action.toast.id ? { ...t, ...action.toast } : t
        ),
      }

    case "DISMISS_TOAST": {
      const { toastId } = action

      // ! Side effects ! - This could be extracted into a dismissToast() action,
      // but I'll keep it here for simplicity
      if (toastId) {
        addToRemoveQueue(toastId)
      } else {
        state.toasts.forEach((toast) => {
          addToRemoveQueue(toast.id)
        })
      }

      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === toastId || toastId === undefined
            ? {
                ...t,
                open: false,
              }
            : t
        ),
      }
    }
    case "REMOVE_TOAST":
      if (action.toastId === undefined) {
        return {
          ...state,
          toasts: [],
        }
      }
      return {
        ...state,
        toasts: state.toasts.filter((t) => t.id !== action.toastId),
      }
  }
}

const listeners: Array<(state: State) => void> = []

let memoryState: State = { toasts: [] }

function dispatch(action: Action) {
  memoryState = reducer(memoryState, action)
  listeners.forEach((listener) => {
    listener(memoryState)
  })
}

type Toast = Omit<ToasterToast, "id">

function toast({ ...props }: Toast) {
  const id = genId()

  const update = (props: ToasterToast) =>
    dispatch({
      type: "UPDATE_TOAST",
      toast: { ...props, id },
    })
  const dismiss = () => dispatch({ type: "DISMISS_TOAST", toastId: id })

  dispatch({
    type: "ADD_TOAST",
    toast: {
      ...props,
      id,
      open: true,
      onOpenChange: (open) => {
        if (!open) dismiss()
      },
    },
  })

  return {
    id: id,
    dismiss,
    update,
  }
}

function useToast() {
  const [state, setState] = React.useState<State>(memoryState)

  React.useEffect(() => {
    listeners.push(setState)
    return () => {
      const index = listeners.indexOf(setState)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }, [state])

  return {
    ...state,
    toast,
    dismiss: (toastId?: string) => dispatch({ type: "DISMISS_TOAST", toastId }),
  }
}

export { useToast, toast }

```

# client\src\index.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 5%;
    --foreground: 0 0% 100%;

    --card: 0 0% 7%;
    --card-foreground: 0 0% 100%;

    --popover: 0 0% 7%;
    --popover-foreground: 0 0% 100%;

    --primary: 330 100% 50%;
    --primary-foreground: 0 0% 100%;

    --secondary: 187 100% 56%;
    --secondary-foreground: 0 0% 100%;

    --muted: 0 0% 16%;
    --muted-foreground: 0 0% 60%;

    --accent: 315 100% 61%;
    --accent-foreground: 0 0% 100%;

    --destructive: 0 100% 50%;
    --destructive-foreground: 0 0% 100%;

    --border: 0 0% 12%;
    --input: 0 0% 12%;
    --ring: 330 100% 50%;

    --radius: 0.5rem;

    --chart-1: 330 100% 50%;
    --chart-2: 187 100% 56%;
    --chart-3: 315 100% 61%;
    --chart-4: 60 100% 50%;
    --chart-5: 120 70% 40%;
    
    --sidebar-background: 0 0% 7%;
    --sidebar-foreground: 0 0% 100%;
    --sidebar-primary: 330 100% 50%;
    --sidebar-primary-foreground: 0 0% 100%;
    --sidebar-accent: 187 100% 56%;
    --sidebar-accent-foreground: 0 0% 100%;
    --sidebar-border: 0 0% 12%;
    --sidebar-ring: 330 100% 50%;
  }
}

@layer base {
  * {
    @apply border-border;
  }

  body {
    @apply bg-background text-foreground font-sans antialiased;
  }
}

/* Custom scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  @apply bg-muted;
}
::-webkit-scrollbar-thumb {
  @apply bg-primary rounded;
}

/* Custom animations */
@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(255, 0, 140, 0.7);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(255, 0, 140, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 0, 140, 0);
  }
}

.pulse {
  animation: pulse 2s infinite;
}

@keyframes loading {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.loading-bar {
  animation: loading 1.5s infinite;
}

@font-face {
  font-family: 'Inter';
  src: url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
}

@font-face {
  font-family: 'Roboto Mono';
  src: url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600&display=swap');
}

.font-mono {
  font-family: 'Roboto Mono', monospace;
}

.font-inter {
  font-family: 'Inter', sans-serif;
}

```

# client\src\lib\api.ts

```ts
import axios from 'axios';
import { useQuery, QueryFunction } from '@tanstack/react-query';
import { 
  ServiceStatus, 
  MemoryStats, 
  NeuralMemoryStatus, 
  NeuralMemoryDiagnostics,
  CCEMetrics,
  Assembly,
  Alert,
  CCEConfig,
  ExplainActivationResponse,
  ExplainMergeResponse,
  LineageResponse,
  MergeLogResponse,
  RuntimeConfigResponse
} from '@shared/schema';

const api = axios.create({
  baseURL: '/api'
});

const defaultQueryFn = async <TData>({ queryKey }: { queryKey: readonly unknown[] }): Promise<TData> => {
  let url = '';
  const params: Record<string, any> = {};
  queryKey.forEach(part => {
    if (typeof part === 'string') {
      url += `/${part}`;
    } else if (typeof part === 'object' && part !== null) {
      Object.assign(params, part);
    }
  });
  if (url.startsWith('/')) {
    url = url.substring(1);
  }
  try {
    const { data } = await api.get(url, { params });
    return data as TData;
  } catch (error: any) {
    console.error(`API Query Error for ${url}:`, error.response?.data || error.message);
    throw new Error(error.response?.data?.message || error.message || `Failed to fetch ${url}`);
  }
};

export const useMemoryCoreHealth = () => {
  return useQuery<ServiceStatus>({
    queryKey: ['memory-core', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatus>({ queryKey: ['memory-core', 'health'] }),
    refetchInterval: false, 
    retry: 2
  });
};

export const useNeuralMemoryHealth = () => {
  return useQuery<ServiceStatus>({
    queryKey: ['neural-memory', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatus>({ queryKey: ['neural-memory', 'health'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEHealth = () => {
  return useQuery<ServiceStatus>({
    queryKey: ['cce', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatus>({ queryKey: ['cce', 'health'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useMemoryCoreStats = () => {
  return useQuery<MemoryStats>({
    queryKey: ['memory-core', 'stats'],
    queryFn: () => defaultQueryFn<MemoryStats>({ queryKey: ['memory-core', 'stats'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAssemblies = () => {
  return useQuery<{ assemblies: Assembly[] }>({
    queryKey: ['memory-core', 'assemblies'],
    queryFn: () => defaultQueryFn<{ assemblies: Assembly[] }>({ queryKey: ['memory-core', 'assemblies'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAssembly = (id: string | null) => {
  return useQuery<Assembly>({
    queryKey: ['memory-core', 'assemblies', id],
    queryFn: () => defaultQueryFn<Assembly>({ queryKey: ['memory-core', 'assemblies', id] }),
    enabled: !!id,
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryStatus = () => {
  return useQuery<NeuralMemoryStatus>({
    queryKey: ['neural-memory', 'status'],
    queryFn: () => defaultQueryFn<NeuralMemoryStatus>({ queryKey: ['neural-memory', 'status'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryDiagnostics = (window: string = '24h') => {
  return useQuery<NeuralMemoryDiagnostics>({
    queryKey: ['neural-memory', 'diagnose_emoloop', { window }],
    queryFn: () => defaultQueryFn<NeuralMemoryDiagnostics>({ queryKey: ['neural-memory', 'diagnose_emoloop', { window }] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEStatus = () => {
  return useQuery<Record<string, any>>({
    queryKey: ['cce', 'status'],
    queryFn: () => defaultQueryFn<Record<string, any>>({ queryKey: ['cce', 'status'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useRecentCCEResponses = () => {
  return useQuery<CCEMetrics>({
    queryKey: ['cce', 'metrics', 'recent_cce_responses'],
    queryFn: () => defaultQueryFn<CCEMetrics>({ queryKey: ['cce', 'metrics', 'recent_cce_responses'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryConfig = () => {
  return useQuery<Record<string, any>>({
    queryKey: ['neural-memory', 'config'],
    queryFn: () => defaultQueryFn<Record<string, any>>({ queryKey: ['neural-memory', 'config'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEConfig = () => {
  return useQuery<CCEConfig>({
    queryKey: ['cce', 'config'],
    queryFn: () => defaultQueryFn<CCEConfig>({ queryKey: ['cce', 'config'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAlerts = () => {
  return useQuery<{success: boolean; data: Alert[]}>({
    queryKey: ['alerts'],
    queryFn: () => defaultQueryFn<{success: boolean; data: Alert[]}>({ queryKey: ['alerts'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useExplainActivation = (assemblyId: string | null, memoryId?: string | null) => {
  const queryParams = memoryId ? { memory_id: memoryId } : {};
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_activation', queryParams] as const;
  return useQuery<ExplainActivationResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<ExplainActivationResponse>({ queryKey }),
    enabled: false, 
    retry: 1,
    staleTime: Infinity,
  });
};

export const useExplainMerge = (assemblyId: string | null) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_merge'] as const;
  return useQuery<ExplainMergeResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<ExplainMergeResponse>({ queryKey }),
    enabled: false, 
    retry: 1,
    staleTime: Infinity,
  });
};

export const useAssemblyLineage = (assemblyId: string | null) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'lineage'] as const;
  return useQuery<LineageResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<LineageResponse>({ queryKey }),
    enabled: !!assemblyId, 
    retry: 1,
    staleTime: 5 * 60 * 1000, 
  });
};

export const useMergeLog = (limit: number = 50) => {
  const queryKey = ['memory-core', 'diagnostics', 'merge_log', { limit }] as const;
  return useQuery<MergeLogResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<MergeLogResponse>({ queryKey }),
    refetchInterval: 30000, 
    staleTime: 15000, 
  });
};

export const useRuntimeConfig = (serviceName: string | null) => {
  const queryKey = ['memory-core', 'config', 'runtime', serviceName] as const;
  return useQuery<RuntimeConfigResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<RuntimeConfigResponse>({ queryKey }),
    enabled: !!serviceName,
    staleTime: 10 * 60 * 1000, 
  });
};

export const verifyMemoryCoreIndex = async () => {
  return api.post('/memory-core/admin/verify_index');
};

export const triggerMemoryCoreRetryLoop = async () => {
  return api.post('/memory-core/admin/trigger_retry_loop');
};

export const initializeNeuralMemory = async () => {
  return api.post('/neural-memory/init');
};

export const setCCEVariant = async (variant: string) => {
  return api.post('/cce/set_variant', { variant });
};

export const refreshAllData = async (queryClient: any) => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'stats'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'assemblies'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'status'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'diagnose_emoloop'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'status'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'metrics', 'recent_cce_responses'] }),
    queryClient.invalidateQueries({ queryKey: ['alerts'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'config', 'runtime'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'diagnostics', 'merge_log'] })
  ]);
};

```

# client\src\lib\api\hooks\useAssemblyDetails.ts

```ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Assembly } from '@shared/schema';

/**
 * Hook to fetch the details of a specific assembly
 */
export const useAssemblyDetails = (assemblyId: string | null | undefined) => {
  return useQuery<Assembly>({
    queryKey: ['/api/memory-core/assemblies/details', assemblyId],
    queryFn: async () => {
      if (!assemblyId) {
        throw new Error('Assembly ID is required');
      }
      const response = await axios.get(`/api/memory-core/assemblies/${assemblyId}`);
      return response.data;
    },
    enabled: !!assemblyId,
    retry: 2,
  });
};

```

# client\src\lib\queryClient.ts

```ts
import { QueryClient, QueryFunction } from "@tanstack/react-query";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const res = await fetch(url, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const res = await fetch(queryKey[0] as string, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});

```

# client\src\lib\store.ts

```ts
import { create } from 'zustand';
import { queryClient } from './queryClient';
import { refreshAllData } from './api';

interface PollingStoreState {
  pollingRate: number;
  pollingInterval: number | null;
  isPolling: boolean;
  setPollingRate: (rate: number) => void;
  startPolling: () => void;
  stopPolling: () => void;
  refreshAllData: () => void;
}

export const usePollingStore = create<PollingStoreState>((set, get) => ({
  pollingRate: 5000, // Default polling rate: 5 seconds
  pollingInterval: null,
  isPolling: false,
  
  setPollingRate: (rate: number) => {
    set({ pollingRate: rate });
    
    // Restart polling with new rate if currently active
    if (get().isPolling) {
      get().stopPolling();
      get().startPolling();
    }
  },
  
  startPolling: () => {
    const { pollingInterval, pollingRate } = get();
    
    // Don't start another interval if one is already running
    if (pollingInterval !== null) {
      return;
    }
    
    // Create new polling interval
    const interval = window.setInterval(() => {
      get().refreshAllData();
    }, pollingRate);
    
    set({ pollingInterval: interval, isPolling: true });
  },
  
  stopPolling: () => {
    const { pollingInterval } = get();
    
    if (pollingInterval !== null) {
      clearInterval(pollingInterval);
      set({ pollingInterval: null, isPolling: false });
    }
  },
  
  refreshAllData: async () => {
    await refreshAllData(queryClient);
  }
}));

interface ThemeStore {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

export const useThemeStore = create<ThemeStore>((set) => ({
  isDarkMode: true, // Default to dark mode for this dashboard
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode }))
}));

interface SidebarStore {
  isCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useSidebarStore = create<SidebarStore>((set) => ({
  isCollapsed: false,
  toggleSidebar: () => set((state) => ({ isCollapsed: !state.isCollapsed }))
}));

```

# client\src\lib\utils.ts

```ts
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { formatDistanceToNow } from "date-fns"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a date as a relative time (e.g. "2 hours ago")
 */
export function formatTimeAgo(dateString: string | null | undefined): string {
  if (!dateString) return 'Unknown';
  try {
    const date = new Date(dateString);
    return formatDistanceToNow(date, { addSuffix: true });
  } catch (error) {
    console.error('Error formatting date:', dateString, error);
    return 'Invalid date';
  }
}

```

# client\src\main.tsx

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";

// Load external fonts
const fontLinks = document.createElement("link");
fontLinks.rel = "stylesheet";
fontLinks.href = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap";
document.head.appendChild(fontLinks);

// Load Font Awesome for icons
const fontAwesome = document.createElement("link");
fontAwesome.rel = "stylesheet";
fontAwesome.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css";
document.head.appendChild(fontAwesome);

// Set page title
const titleElement = document.createElement("title");
titleElement.textContent = "Synthians Cognitive Dashboard";
document.head.appendChild(titleElement);

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);

```

# client\src\pages\admin.tsx

```tsx
import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { verifyMemoryCoreIndex, triggerMemoryCoreRetryLoop, initializeNeuralMemory, setCCEVariant } from "@/lib/api";

export default function Admin() {
  const { toast } = useToast();
  const [selectedVariant, setSelectedVariant] = useState("MAC-13b");
  const [isLoading, setIsLoading] = useState({
    verifyIndex: false,
    retryLoop: false,
    initNM: false,
    setVariant: false
  });
  const [lastActionResult, setLastActionResult] = useState<{
    action: string;
    success: boolean;
    message: string;
  } | null>(null);
  
  // Handle verify index action
  const handleVerifyIndex = async () => {
    setIsLoading({ ...isLoading, verifyIndex: true });
    try {
      await verifyMemoryCoreIndex();
      toast({
        title: "Success",
        description: "Index verification triggered successfully",
      });
      setLastActionResult({
        action: "Verify Memory Core Index",
        success: true,
        message: "Index verification has been triggered. This process will run in the background."
      });
    } catch (error) {
      console.error("Failed to verify index:", error);
      toast({
        title: "Error",
        description: "Failed to trigger index verification",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Verify Memory Core Index",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, verifyIndex: false });
    }
  };
  
  // Handle retry loop action
  const handleRetryLoop = async () => {
    setIsLoading({ ...isLoading, retryLoop: true });
    try {
      await triggerMemoryCoreRetryLoop();
      toast({
        title: "Success",
        description: "Retry loop triggered successfully",
      });
      setLastActionResult({
        action: "Trigger Retry Loop",
        success: true,
        message: "Retry loop has been triggered. Pending operations will be reprocessed."
      });
    } catch (error) {
      console.error("Failed to trigger retry loop:", error);
      toast({
        title: "Error",
        description: "Failed to trigger retry loop",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Trigger Retry Loop",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, retryLoop: false });
    }
  };
  
  // Handle Neural Memory initialization
  const handleInitializeNM = async () => {
    setIsLoading({ ...isLoading, initNM: true });
    try {
      await initializeNeuralMemory();
      toast({
        title: "Success",
        description: "Neural Memory initialized successfully",
      });
      setLastActionResult({
        action: "Initialize Neural Memory",
        success: true,
        message: "Neural Memory module has been reinitialized."
      });
    } catch (error) {
      console.error("Failed to initialize Neural Memory:", error);
      toast({
        title: "Error",
        description: "Failed to initialize Neural Memory",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Initialize Neural Memory",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, initNM: false });
    }
  };
  
  // Handle CCE variant selection
  const handleSetVariant = async () => {
    setIsLoading({ ...isLoading, setVariant: true });
    try {
      await setCCEVariant(selectedVariant);
      toast({
        title: "Success",
        description: `Variant set to ${selectedVariant} successfully`,
      });
      setLastActionResult({
        action: "Set CCE Variant",
        success: true,
        message: `CCE Variant has been set to ${selectedVariant}. This will affect future responses.`
      });
    } catch (error) {
      console.error("Failed to set CCE variant:", error);
      toast({
        title: "Error",
        description: "Failed to set CCE variant",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Set CCE Variant",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, setVariant: false });
    }
  };

  return (
    <>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white mb-1">Admin Actions</h2>
        <p className="text-sm text-gray-400">
          Manually trigger maintenance tasks for testing and debugging
        </p>
      </div>
      
      <Alert className="mb-6 border-yellow-600 bg-yellow-950/30">
        <i className="fas fa-exclamation-triangle text-yellow-400 mr-2"></i>
        <AlertTitle>Warning: Administrative Area</AlertTitle>
        <AlertDescription>
          These actions can affect the performance and behavior of the Synthians Cognitive Architecture services.
          Use with caution in production environments.
        </AlertDescription>
      </Alert>
      
      {lastActionResult && (
        <Alert 
          className={`mb-6 ${lastActionResult.success 
            ? "border-green-600 bg-green-950/30" 
            : "border-red-600 bg-red-950/30"}`}
        >
          <i className={`fas ${lastActionResult.success ? "fa-check-circle text-green-400" : "fa-times-circle text-red-400"} mr-2`}></i>
          <AlertTitle>{lastActionResult.action} - {lastActionResult.success ? "Success" : "Failed"}</AlertTitle>
          <AlertDescription>
            {lastActionResult.message}
          </AlertDescription>
        </Alert>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Memory Core Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Memory Core Actions</CardTitle>
            <CardDescription>Maintenance operations for the Memory Core service</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-2">Verify Vector Index</h3>
              <p className="text-sm text-gray-400 mb-4">
                Triggers a background job to verify the integrity of the vector index.
                This will compare indexed vectors against their source memories.
              </p>
              <Button 
                onClick={handleVerifyIndex} 
                disabled={isLoading.verifyIndex}
                className="w-full"
              >
                {isLoading.verifyIndex && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Verify Index
              </Button>
            </div>
            
            <Separator />
            
            <div>
              <h3 className="text-sm font-medium mb-2">Trigger Retry Loop</h3>
              <p className="text-sm text-gray-400 mb-4">
                Forces the Memory Core to retry any pending or failed operations,
                such as vector updates or assembly indexing.
              </p>
              <Button 
                onClick={handleRetryLoop} 
                disabled={isLoading.retryLoop}
                className="w-full"
              >
                {isLoading.retryLoop && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Trigger Retry Loop
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Neural Memory Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Neural Memory Actions</CardTitle>
            <CardDescription>Operations for the Neural Memory module</CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <h3 className="text-sm font-medium mb-2">Initialize Neural Memory</h3>
              <p className="text-sm text-gray-400 mb-4">
                Reinitializes the Neural Memory module, resetting its internal state.
                This is useful if the module becomes unstable or unresponsive.
              </p>
              <Button 
                onClick={handleInitializeNM} 
                disabled={isLoading.initNM}
                variant="destructive"
                className="w-full"
              >
                {isLoading.initNM && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Reset Neural Memory
              </Button>
              <p className="text-xs text-destructive mt-2">
                <i className="fas fa-exclamation-circle mr-1"></i>
                Warning: This will reset any in-progress emotional loop training.
              </p>
            </div>
          </CardContent>
        </Card>
        
        {/* CCE Actions */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Context Cascade Engine Actions</CardTitle>
            <CardDescription>Control operations for the CCE service</CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <h3 className="text-sm font-medium mb-2">Set CCE Variant</h3>
              <p className="text-sm text-gray-400 mb-4">
                Manually override the active variant used by the Context Cascade Engine.
                This will bypass the automatic selection mechanism.
              </p>
              <div className="flex space-x-4">
                <Select value={selectedVariant} onValueChange={setSelectedVariant}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select variant" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MAC-7b">MAC-7b</SelectItem>
                    <SelectItem value="MAC-13b">MAC-13b</SelectItem>
                    <SelectItem value="TITAN-7b">TITAN-7b</SelectItem>
                  </SelectContent>
                </Select>
                
                <Button 
                  onClick={handleSetVariant} 
                  disabled={isLoading.setVariant}
                  className="flex-1"
                >
                  {isLoading.setVariant && <i className="fas fa-spin fa-spinner mr-2"></i>}
                  Set Variant to {selectedVariant}
                </Button>
              </div>
            </div>
          </CardContent>
          <CardFooter className="bg-muted/50 flex justify-between">
            <p className="text-xs text-gray-400">
              <i className="fas fa-info-circle mr-1"></i>
              These endpoints may respond with a 501 Not Implemented if the backend service does not support them yet.
            </p>
            
            <Button variant="ghost" size="sm" onClick={() => setLastActionResult(null)}>
              Clear Results
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  );
}

```

# client\src\pages\assemblies\[id].tsx

```tsx
import React, { useEffect, useState } from "react";
import { useAssembly } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { useToast } from "@/hooks/use-toast";
import { Link, useParams } from "wouter";
import { usePollingStore } from "@/lib/store";

export default function AssemblyDetail() {
  const { id } = useParams();
  const { refreshAllData } = usePollingStore();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Fetch assembly data
  const { data, isLoading, isError, error, refetch } = useAssembly(id || null);
  
  useEffect(() => {
    if (isError) {
      toast({
        title: "Error loading assembly",
        description: (error as Error)?.message || "Could not load assembly details",
        variant: "destructive"
      });
    }
  }, [isError, error, toast]);
  
  // Helper function to get sync status
  const getSyncStatus = (assembly: any) => {
    if (!assembly?.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-400",
        bgColor: "bg-muted/50"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-secondary",
        bgColor: "bg-muted/50"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-primary",
      bgColor: "bg-muted/50"
    };
  };
  
  // Format date for display
  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };
  
  const assembly = data?.data || null;
  const syncStatus = assembly ? getSyncStatus(assembly) : null;
  
  const handleRefresh = () => {
    refetch();
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <div className="flex items-center mb-1">
            <Link href="/assemblies">
              <Button variant="ghost" size="sm" className="mr-2 -ml-2">
                <i className="fas fa-arrow-left mr-1"></i> Back
              </Button>
            </Link>
            <h2 className="text-xl font-semibold text-white">Assembly Inspector</h2>
          </div>
          <p className="text-sm text-gray-400">
            {isLoading ? (
              <Skeleton className="h-4 w-64 inline-block" />
            ) : assembly ? (
              <>Viewing details for assembly <code className="text-primary">{assembly.id}</code></>
            ) : (
              <>Assembly not found</>
            )}
          </p>
        </div>
        <RefreshButton onClick={handleRefresh} />
      </div>
      
      {isLoading ? (
        <div className="space-y-6">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      ) : !assembly ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <i className="fas fa-folder-open text-4xl text-muted-foreground mb-4"></i>
            <h3 className="text-xl font-medium mb-2">Assembly Not Found</h3>
            <p className="text-muted-foreground mb-6">The assembly with ID "{id}" could not be found.</p>
            <Link href="/assemblies">
              <Button>
                <i className="fas fa-arrow-left mr-2"></i> Back to Assemblies
              </Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Assembly Header */}
          <Card className="mb-6">
            <CardHeader>
              <div className="flex justify-between">
                <div>
                  <CardTitle className="text-xl">{assembly.name}</CardTitle>
                  <CardDescription className="mt-2">{assembly.description || "No description provided"}</CardDescription>
                </div>
                <Badge 
                  variant="outline" 
                  className={`${syncStatus?.bgColor} ${syncStatus?.color} self-start`}
                >
                  {syncStatus?.label}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Assembly ID</p>
                  <p className="font-mono text-secondary">{assembly.id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Memory Count</p>
                  <p className="font-mono">{assembly.member_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Created</p>
                  <p className="text-sm">{formatDate(assembly.created_at)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Last Updated</p>
                  <p className="text-sm">{formatDate(assembly.updated_at)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Tabs for Content */}
          <Tabs defaultValue="overview" className="mb-6">
            <TabsList>
              <TabsTrigger value="overview" onClick={() => setActiveTab("overview")}>Overview</TabsTrigger>
              <TabsTrigger value="members" onClick={() => setActiveTab("members")}>Memory Members</TabsTrigger>
              <TabsTrigger value="metadata" onClick={() => setActiveTab("metadata")}>Metadata</TabsTrigger>
              {assembly.vector_index_updated_at && (
                <TabsTrigger value="embedding" onClick={() => setActiveTab("embedding")}>Embedding</TabsTrigger>
              )}
            </TabsList>
            
            <TabsContent value="overview" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Assembly Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Key Information</h3>
                      
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Name</p>
                          <p>{assembly.name}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Description</p>
                          <p>{assembly.description || "No description available"}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Member Count</p>
                          <p>{assembly.member_count} memories</p>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-4">Sync Status</h3>
                      
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Last Updated</p>
                          <p>{formatDate(assembly.updated_at)}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Vector Index Updated</p>
                          <p>{assembly.vector_index_updated_at ? formatDate(assembly.vector_index_updated_at) : "Not indexed yet"}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Status</p>
                          <Badge 
                            variant="outline" 
                            className={`${syncStatus?.bgColor} ${syncStatus?.color}`}
                          >
                            {syncStatus?.label}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="members" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Memory Members</CardTitle>
                  <CardDescription>List of memory IDs that are part of this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  {assembly.memory_ids && assembly.memory_ids.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-96 overflow-y-auto p-1">
                      {assembly.memory_ids.map((memoryId: string) => (
                        <div key={memoryId} className="bg-muted p-2 rounded-md font-mono text-xs flex items-center">
                          <i className="fas fa-memory text-secondary mr-2"></i>
                          {memoryId}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      <i className="fas fa-info-circle mr-2"></i>
                      No memory members found
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="metadata" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Metadata</CardTitle>
                  <CardDescription>Keywords, tags, and topics associated with this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h3 className="text-sm font-medium mb-3">Keywords</h3>
                      {assembly.keywords && assembly.keywords.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.keywords.map((keyword: string, idx: number) => (
                            <Badge key={idx} variant="secondary">
                              {keyword}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No keywords available</p>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-3">Tags</h3>
                      {assembly.tags && assembly.tags.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.tags.map((tag: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="border-primary text-primary">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No tags available</p>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-3">Topics</h3>
                      {assembly.topics && assembly.topics.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.topics.map((topic: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="border-secondary text-secondary">
                              {topic}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No topics available</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            {assembly.vector_index_updated_at && (
              <TabsContent value="embedding" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Composite Embedding</CardTitle>
                    <CardDescription>Vector representation visualization (placeholder)</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-muted rounded-lg p-6 flex flex-col items-center justify-center min-h-[240px]">
                      <div className="mb-4 text-center">
                        <i className="fas fa-project-diagram text-4xl text-primary mb-4"></i>
                        <p className="text-muted-foreground">
                          Embedding visualization is a future enhancement.
                        </p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-md">
                        <div className="bg-card p-3 rounded">
                          <p className="text-xs text-gray-500 mb-1">Embedding Norm</p>
                          <p className="font-mono">0.9873</p>
                        </div>
                        <div className="bg-card p-3 rounded">
                          <p className="text-xs text-gray-500 mb-1">Sparsity</p>
                          <p className="font-mono">0.0418</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            )}
          </Tabs>
        </>
      )}
    </>
  );
}

```

# client\src\pages\assemblies\assembly-inspector.tsx

```tsx
import React, { useState } from 'react';
import { useLocation, useParams } from 'wouter';
import { useAssemblyDetails } from '@/lib/api/hooks/useAssemblyDetails';
import { useAssemblyLineage, useExplainMerge, useExplainActivation } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineageView } from '@/components/dashboard/LineageView';
import { MergeExplanationView } from '@/components/dashboard/MergeExplanationView';
import { ActivationExplanationView } from '@/components/dashboard/ActivationExplanationView';
import { ArrowLeft, Calendar, Tag } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useFeatures } from '@/contexts/FeaturesContext';
import { ExplainMergeData, ExplainActivationData } from '@shared/schema';

export default function AssemblyInspector() {
  const [, setLocation] = useLocation();
  const params = useParams<{ id: string }>();
  const assemblyId = params?.id;
  const { explainabilityEnabled } = useFeatures();
  
  // Selected memory for activation explanation
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  
  // Fetch assembly details
  const { data: assembly, isLoading, isError, error } = useAssemblyDetails(assemblyId);
  
  // Fetch lineage data
  const lineageQuery = useAssemblyLineage(assemblyId);
  
  // Prepare merge explanation query (triggered manually)
  const mergeExplanationQuery = useExplainMerge(assemblyId);
  
  // Prepare activation explanation query (triggered manually)
  const activationExplanationQuery = useExplainActivation(assemblyId, selectedMemoryId);
  
  // Handle memory selection for activation explanation
  const handleMemorySelect = (memoryId: string) => {
    setSelectedMemoryId(memoryId);
    activationExplanationQuery.refetch();
  };
  
  // Handle loading merge explanation
  const handleExplainMerge = () => {
    mergeExplanationQuery.refetch();
  };
  
  if (isLoading) {
    return (
      <div className="container py-6">
        <div className="space-y-6">
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
          </div>
          <Card>
            <CardHeader>
              <CardTitle><Skeleton className="h-6 w-48" /></CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-2/3" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }
  
  if (isError || !assembly) {
    return (
      <div className="container py-6">
        <div className="space-y-6">
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
          </div>
          <Card>
            <CardContent className="p-6">
              <div className="text-center py-8">
                <h2 className="text-xl font-medium text-red-500 mb-2">Failed to load assembly details</h2>
                <p className="text-muted-foreground">
                  {error?.message || 'Could not retrieve assembly information. Please try again.'}
                </p>
                <Button className="mt-4" onClick={() => setLocation('/assemblies')}>
                  Return to Assemblies
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container py-6">
      <div className="space-y-6">
        {/* Header with back button */}
        <div className="flex items-center">
          <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
        </div>
        
        {/* Assembly information card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{assembly.name || 'Unnamed Assembly'}</span>
              <Badge variant="outline">{assembly.id}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">{assembly.description || 'No description available'}</p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h3 className="text-sm font-medium flex items-center gap-1 text-muted-foreground">
                  <Calendar className="h-4 w-4" /> Created
                </h3>
                <p>
                  {new Date(assembly.created_at).toLocaleString()} 
                  <span className="text-muted-foreground ml-2 text-sm">
                    ({formatDistanceToNow(new Date(assembly.created_at), { addSuffix: true })})
                  </span>
                </p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium flex items-center gap-1 text-muted-foreground">
                  <Tag className="h-4 w-4" /> Tags
                </h3>
                <div className="flex flex-wrap gap-1 mt-1">
                  {assembly.tags && assembly.tags.length > 0 ? (
                    assembly.tags.map((tag: string) => (
                      <Badge key={tag} variant="secondary">{tag}</Badge>
                    ))
                  ) : (
                    <span className="text-muted-foreground text-sm">No tags</span>
                  )}
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Memories</h3>
                <p>{assembly.memory_ids?.length || 0} memories in this assembly</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {!explainabilityEnabled && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 my-2">
            <p className="font-medium">Explainability features are disabled</p>
            <p className="text-sm">Some features like merge explanations and activation details are not available. Enable them by setting <code>ENABLE_EXPLAINABILITY=true</code> in the Memory Core configuration.</p>
          </div>
        )}
        
        {/* Explainability tabs */}
        <Tabs defaultValue="lineage" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="lineage">Lineage</TabsTrigger>
            <TabsTrigger value="merge" disabled={!explainabilityEnabled}>Merge Explanation</TabsTrigger>
            <TabsTrigger value="memories" disabled={!explainabilityEnabled}>Memories & Activation</TabsTrigger>
          </TabsList>
          
          <TabsContent value="lineage" className="pt-4">
            <LineageView 
              lineage={lineageQuery.data?.lineage} 
              isLoading={lineageQuery.isLoading} 
              isError={lineageQuery.isError} 
              error={lineageQuery.error as Error}
            />
          </TabsContent>
          
          <TabsContent value="merge" className="pt-4">
            <div className="space-y-4">
              {!mergeExplanationQuery.data && !mergeExplanationQuery.isLoading && (
                <div className="text-center p-6 bg-muted rounded-md">
                  <p className="mb-4">Merge explanation data hasn't been loaded yet.</p>
                  <Button onClick={handleExplainMerge} disabled={!explainabilityEnabled}>
                    Explain How This Assembly Was Formed
                  </Button>
                </div>
              )}
              
              {(mergeExplanationQuery.data || mergeExplanationQuery.isLoading) && (
                <MergeExplanationView 
                  mergeData={mergeExplanationQuery.data?.explanation as ExplainMergeData | undefined} 
                  isLoading={mergeExplanationQuery.isLoading} 
                  isError={mergeExplanationQuery.isError} 
                  error={mergeExplanationQuery.error as Error}
                />
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="memories" className="pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Memory selection */}
              <Card>
                <CardHeader>
                  <CardTitle>Memories in Assembly</CardTitle>
                </CardHeader>
                <CardContent>
                  {assembly.memory_ids?.length === 0 ? (
                    <div className="text-center p-4 text-muted-foreground">
                      <p>No memories in this assembly.</p>
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                      {assembly.memory_ids?.map((memoryId: string) => (
                        <div 
                          key={memoryId}
                          className={`p-3 border rounded-md cursor-pointer hover:bg-muted transition-colors ${selectedMemoryId === memoryId ? 'border-primary bg-primary/5' : ''}`}
                          onClick={() => handleMemorySelect(memoryId)}
                        >
                          <p className="font-medium truncate">{memoryId}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {/* Activation explanation */}
              <div>
                {!selectedMemoryId ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Memory Activation Details</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center p-4 text-muted-foreground">
                        <p>Select a memory to see activation details.</p>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <ActivationExplanationView 
                    activationData={activationExplanationQuery.data?.explanation as ExplainActivationData | undefined} 
                    memoryId={selectedMemoryId}
                    isLoading={activationExplanationQuery.isLoading} 
                    isError={activationExplanationQuery.isError} 
                    error={activationExplanationQuery.error as Error}
                  />
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

```

# client\src\pages\assemblies\index.tsx

```tsx
import React, { useState } from "react";
import { useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { Link } from "wouter";
import { usePollingStore } from "@/lib/store";

export default function AssembliesIndex() {
  const { refreshAllData } = usePollingStore();
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("updated");
  const [sortOrder, setSortOrder] = useState("desc");
  const [statusFilter, setStatusFilter] = useState("all");
  
  // Fetch assemblies data
  const { data, isLoading, isError } = useAssemblies();
  
  // Filter and sort assemblies
  const filteredAssemblies = React.useMemo(() => {
    if (!data?.data) return [];
    
    let filtered = [...data.data];
    
    // Apply search filter
    if (searchTerm) {
      const lowercaseTerm = searchTerm.toLowerCase();
      filtered = filtered.filter(assembly => 
        assembly.id.toLowerCase().includes(lowercaseTerm) ||
        assembly.name.toLowerCase().includes(lowercaseTerm)
      );
    }
    
    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(assembly => {
        if (!assembly.vector_index_updated_at) {
          return statusFilter === "pending";
        }
        
        const vectorDate = new Date(assembly.vector_index_updated_at);
        const updateDate = new Date(assembly.updated_at);
        
        if (statusFilter === "indexed") {
          return vectorDate >= updateDate;
        } else if (statusFilter === "syncing") {
          return vectorDate < updateDate;
        }
        
        return true;
      });
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      if (sortBy === "id") {
        aValue = a.id;
        bValue = b.id;
      } else if (sortBy === "name") {
        aValue = a.name;
        bValue = b.name;
      } else if (sortBy === "members") {
        aValue = a.member_count;
        bValue = b.member_count;
      } else if (sortBy === "updated") {
        aValue = new Date(a.updated_at).getTime();
        bValue = new Date(b.updated_at).getTime();
      }
      
      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
    
    return filtered;
  }, [data, searchTerm, sortBy, sortOrder, statusFilter]);
  
  // Helper function to get sync status
  const getSyncStatus = (assembly: any) => {
    if (!assembly.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-400",
        bgColor: "bg-muted/50"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-secondary",
        bgColor: "bg-muted/50"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-primary",
      bgColor: "bg-muted/50"
    };
  };
  
  // Format time ago
  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    
    if (diffMin < 60) {
      return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
    } else if (diffMin < 1440) {
      const hours = Math.floor(diffMin / 60);
      return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    } else {
      const days = Math.floor(diffMin / 1440);
      return `${days} day${days === 1 ? '' : 's'} ago`;
    }
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Assembly Inspector</h2>
          <p className="text-sm text-gray-400">Browse and inspect memory assemblies</p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Filter Controls */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Input
                type="text"
                placeholder="Search by ID or name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
              <i className="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"></i>
            </div>
            
            <div className="flex space-x-2">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="id">ID</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="members">Member Count</SelectItem>
                  <SelectItem value="updated">Last Updated</SelectItem>
                </SelectContent>
              </Select>
              
              <Button
                variant="outline"
                size="icon"
                onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              >
                <i className={`fas fa-sort-${sortOrder === "asc" ? "up" : "down"}`}></i>
              </Button>
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="indexed">Indexed</SelectItem>
                <SelectItem value="syncing">Syncing</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
      
      {/* Assemblies Table */}
      <Card>
        <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <CardTitle className="font-medium">Memory Assemblies</CardTitle>
            {!isLoading && (
              <Badge variant="outline" className="ml-2">
                {filteredAssemblies.length} {filteredAssemblies.length === 1 ? 'assembly' : 'assemblies'}
              </Badge>
            )}
          </div>
        </CardHeader>
        
        <div className="overflow-x-auto">
          <Table>
            <TableHeader className="bg-muted">
              <TableRow>
                <TableHead className="w-[180px]">Assembly ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead className="text-center">Member Count</TableHead>
                <TableHead>Last Updated</TableHead>
                <TableHead>Sync Status</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            
            <TableBody>
              {isLoading ? (
                Array(5).fill(0).map((_, index) => (
                  <TableRow key={index}>
                    <TableCell><Skeleton className="h-6 w-28" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-48" /></TableCell>
                    <TableCell className="text-center"><Skeleton className="h-6 w-16 mx-auto" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-12 ml-auto" /></TableCell>
                  </TableRow>
                ))
              ) : isError ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-400">
                    <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                    Failed to load assemblies. Please try again.
                  </TableCell>
                </TableRow>
              ) : filteredAssemblies.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-400">
                    {searchTerm ? (
                      <>
                        <i className="fas fa-search mr-2"></i>
                        No assemblies matching "{searchTerm}"
                      </>
                    ) : (
                      <>
                        <i className="fas fa-info-circle mr-2"></i>
                        No assemblies found
                      </>
                    )}
                  </TableCell>
                </TableRow>
              ) : (
                filteredAssemblies.map((assembly) => {
                  const syncStatus = getSyncStatus(assembly);
                  return (
                    <TableRow key={assembly.id} className="hover:bg-muted">
                      <TableCell className="font-mono text-secondary">{assembly.id}</TableCell>
                      <TableCell className="font-medium">{assembly.name}</TableCell>
                      <TableCell className="text-center">{assembly.member_count}</TableCell>
                      <TableCell className="text-sm text-gray-400">{formatTimeAgo(assembly.updated_at)}</TableCell>
                      <TableCell>
                        <Badge 
                          variant="outline" 
                          className={`${syncStatus.bgColor} ${syncStatus.color}`}
                        >
                          {syncStatus.label}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Link href={`/assemblies/${assembly.id}`}>
                          <Button variant="ghost" size="sm" className="text-primary hover:text-accent text-xs">
                            View <i className="fas fa-chevron-right ml-1"></i>
                          </Button>
                        </Link>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </>
  );
}

```

# client\src\pages\cce.tsx

```tsx
import React, { useState } from "react";
import { useCCEHealth, useCCEStatus, useRecentCCEResponses } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { CCEChart } from "@/components/dashboard/CCEChart";
import { usePollingStore } from "@/lib/store";

export default function CCE() {
  const { refreshAllData } = usePollingStore();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Fetch CCE data
  const cceHealth = useCCEHealth();
  const cceStatus = useCCEStatus();
  const recentCCEResponses = useRecentCCEResponses();
  
  // Prepare service status object
  const serviceStatus = cceHealth.data?.data ? {
    name: "Context Cascade Engine",
    status: cceHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || "Unknown",
    version: cceHealth.data.data.version || "Unknown"
  } : null;
  
  // Get active variant from the most recent response
  const activeVariant = recentCCEResponses.data?.data?.recent_responses?.[0]?.variant_output?.variant_type || "Unknown";
  
  // Filter recent responses with errors
  const errorResponses = recentCCEResponses.data?.data?.recent_responses?.filter(
    response => response.status === "error"
  ) || [];
  
  // Get variant selections for display
  const variantSelections = recentCCEResponses.data?.data?.recent_responses?.filter(
    response => response.variant_selection
  ).slice(0, 10) || [];
  
  // Get responses with LLM guidance
  const llmGuidanceResponses = recentCCEResponses.data?.data?.recent_responses?.filter(
    response => response.llm_advice_used
  ).slice(0, 5) || [];
  
  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">CCE Dashboard</h2>
          <p className="text-sm text-gray-400">
            Monitoring the <code className="text-primary">Context Cascade Engine</code> and variant selection
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {serviceStatus ? (
              <ServiceStatus service={serviceStatus} />
            ) : (
              <Skeleton className="h-5 w-20" />
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
            <div>
              <p className="text-sm text-gray-500 mb-1">Connection</p>
              {cceHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus ? (
                <p className="text-lg">{serviceStatus.url}</p>
              ) : (
                <p className="text-red-500">Unreachable</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Uptime</p>
              {cceHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus?.uptime ? (
                <p className="text-lg">{serviceStatus.uptime}</p>
              ) : (
                <p className="text-gray-400">Unknown</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Active Variant</p>
              {recentCCEResponses.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : (
                <p className="text-lg font-mono text-secondary">{activeVariant}</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Tabs for different views */}
      <Tabs defaultValue="variants" className="mb-6">
        <TabsList>
          <TabsTrigger value="variants" onClick={() => setActiveTab("variants")}>Variant Selection</TabsTrigger>
          <TabsTrigger value="llm" onClick={() => setActiveTab("llm")}>LLM Guidance</TabsTrigger>
          <TabsTrigger value="errors" onClick={() => setActiveTab("errors")}>Errors</TabsTrigger>
        </TabsList>
        
        <TabsContent value="variants" className="mt-4">
          <div className="grid grid-cols-1 gap-6">
            <CCEChart
              title="Variant Distribution (Last 12 Hours)"
              data={recentCCEResponses.data?.data?.recent_responses || []}
              isLoading={recentCCEResponses.isLoading}
            />
            
            <Card>
              <CardHeader>
                <CardTitle>Recent Variant Selections</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-32 w-full" />
                  </div>
                ) : variantSelections.length > 0 ? (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[120px]">Timestamp</TableHead>
                          <TableHead>Selected Variant</TableHead>
                          <TableHead>Reason</TableHead>
                          <TableHead className="text-center">Perf. Used</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {variantSelections.map((response, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-mono text-xs">
                              {formatTime(response.timestamp)}
                            </TableCell>
                            <TableCell>
                              <Badge className="bg-muted text-secondary">
                                {response.variant_selection?.selected_variant}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-sm">
                              {response.variant_selection?.reason || "N/A"}
                            </TableCell>
                            <TableCell className="text-center">
                              {response.variant_selection?.performance_used ? (
                                <i className="fas fa-check text-green-400"></i>
                              ) : (
                                <i className="fas fa-times text-gray-500"></i>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <p className="text-gray-400 text-center py-4">No variant selection data available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="llm" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>LLM Guidance Usage</CardTitle>
            </CardHeader>
            <CardContent>
              {recentCCEResponses.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-64 w-full" />
                </div>
              ) : llmGuidanceResponses.length > 0 ? (
                <div className="space-y-4">
                  {llmGuidanceResponses.map((response, index) => (
                    <div key={index} className="border border-border rounded-lg p-4">
                      <div className="flex justify-between mb-3">
                        <span className="text-xs text-gray-400">
                          {new Date(response.timestamp).toLocaleString()}
                        </span>
                        <Badge variant="outline" className="text-primary border-primary">
                          Confidence: {response.llm_advice_used?.confidence_level.toFixed(2)}
                        </Badge>
                      </div>
                      
                      <h4 className="text-sm font-medium mb-2">Adjusted Advice</h4>
                      <div className="bg-muted p-3 rounded text-sm mb-4 font-mono">
                        {response.llm_advice_used?.adjusted_advice || "N/A"}
                      </div>
                      
                      {response.llm_advice_used?.raw_advice && (
                        <>
                          <h4 className="text-sm font-medium mb-2">Raw LLM Advice</h4>
                          <div className="bg-muted p-3 rounded text-sm mb-4 font-mono text-xs overflow-auto max-h-32">
                            {response.llm_advice_used.raw_advice}
                          </div>
                        </>
                      )}
                      
                      {response.llm_advice_used?.adjustment_reason && (
                        <div className="text-xs text-gray-400">
                          <span className="text-secondary">Adjustment Reason:</span> {response.llm_advice_used.adjustment_reason}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-center py-4">No LLM guidance data available</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="errors" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Errors</CardTitle>
            </CardHeader>
            <CardContent>
              {recentCCEResponses.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-32 w-full" />
                </div>
              ) : errorResponses.length > 0 ? (
                <div className="space-y-4">
                  {errorResponses.map((response, index) => (
                    <div key={index} className="border border-border rounded-lg p-4 bg-red-900/10">
                      <div className="flex items-start">
                        <i className="fas fa-exclamation-circle text-destructive mr-3 mt-1"></i>
                        <div>
                          <div className="flex items-center mb-2">
                            <h4 className="text-sm font-medium mr-2">Error at {formatTime(response.timestamp)}</h4>
                            <Badge variant="destructive">Error</Badge>
                          </div>
                          <p className="text-sm text-gray-300 mb-2">{response.error_details}</p>
                          
                          {response.variant_selection && (
                            <div className="text-xs text-gray-400">
                              <span>Attempted variant: </span>
                              <span className="text-primary">{response.variant_selection.selected_variant}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <i className="fas fa-check-circle text-green-400 text-2xl mb-2"></i>
                  <p className="text-gray-400">No errors detected in recent responses</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}

```

# client\src\pages\chat.tsx

```tsx
import React, { useState, useRef, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  metrics?: {
    variant?: string;
    surprise_level?: number;
    retrieved_memory_ids?: string[];
  };
  isLoading?: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollArea = scrollAreaRef.current;
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }
  }, [messages]);
  
  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    
    // Add user message
    const userMessage: ChatMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    };
    
    setMessages([...messages, userMessage]);
    
    // Add loading state for assistant message
    const loadingMessage: ChatMessage = {
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true
    };
    
    setMessages(prev => [...prev, loadingMessage]);
    
    // Clear input and set typing indicator
    setInputValue("");
    setIsTyping(true);
    
    // This is a placeholder for backend integration
    // In a real implementation, this would call the CCE service
    console.log("Sending message:", inputValue);
    
    // Simulate response after a delay
    setTimeout(() => {
      setIsTyping(false);
      
      // Replace loading message with placeholder response
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages.pop(); // Remove loading message
        
        // Add simulated response
        newMessages.push({
          role: "assistant",
          content: "Chat interface connected. Waiting for input... (Backend integration required)",
          timestamp: new Date(),
          metrics: {
            variant: "MAC-13b",
            surprise_level: 0.42,
            retrieved_memory_ids: ["MEM-123456", "MEM-789012"]
          }
        });
        
        return newMessages;
      });
    }, 1500);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const handleClearChat = () => {
    setMessages([]);
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Chat Interface</h2>
          <p className="text-sm text-gray-400">
            Direct interaction interface with an AI persona powered by the Synthians memory system
          </p>
        </div>
        
        <Button variant="outline" size="sm" onClick={handleClearChat}>
          <i className="fas fa-trash mr-2"></i>
          Clear Chat
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="lg:col-span-3">
          <CardHeader className="pb-2">
            <div className="flex items-center">
              <Avatar className="mr-2 h-8 w-8 bg-primary">
                <span className="text-xs">AI</span>
              </Avatar>
              <div>
                <CardTitle>Synthians Chat</CardTitle>
                <CardDescription>
                  Phase 6 Preparation - Placeholder for end-to-end testing
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {/* Chat Messages */}
            <ScrollArea className="h-[calc(100vh-320px)] p-4" ref={scrollAreaRef}>
              {messages.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <i className="fas fa-comments text-4xl mb-4 text-muted-foreground"></i>
                  <p>Send a message to start the conversation</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div 
                      key={index} 
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div 
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user" 
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        }`}
                      >
                        {message.isLoading ? (
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping"></div>
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping" style={{ animationDelay: "0.2s" }}></div>
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping" style={{ animationDelay: "0.4s" }}></div>
                          </div>
                        ) : (
                          <>
                            <p className="mb-1">{message.content}</p>
                            <div className="text-xs opacity-70 mt-1 flex justify-between">
                              <span>{message.timestamp.toLocaleTimeString()}</span>
                              
                              {message.metrics && (
                                <span className="ml-2">
                                  {message.metrics.variant && (
                                    <Badge variant="outline" className="text-secondary border-secondary mr-1">
                                      {message.metrics.variant}
                                    </Badge>
                                  )}
                                </span>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
            
            {/* Input Area */}
            <div className="p-4 border-t border-border">
              <div className="flex space-x-2">
                <Input
                  placeholder="Type a message..."
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
                <Button onClick={handleSendMessage} disabled={!inputValue.trim()}>
                  <i className="fas fa-paper-plane mr-2"></i>
                  Send
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Metrics Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Interaction Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Active Variant</p>
                <Badge className="bg-muted text-secondary">MAC-13b</Badge>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 mb-1">Recent Surprise Level</p>
                <div className="h-2 w-full bg-muted rounded-full">
                  <div 
                    className="h-2 bg-gradient-to-r from-blue-500 to-primary rounded-full"
                    style={{ width: "42%" }}
                  ></div>
                </div>
                <p className="text-xs mt-1 text-right">0.42</p>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 mb-1">Retrieved Memories</p>
                <div className="bg-muted p-2 rounded text-xs font-mono max-h-32 overflow-y-auto">
                  <div className="flex items-center mb-1">
                    <i className="fas fa-memory text-secondary mr-1"></i>
                    <span>MEM-123456</span>
                  </div>
                  <div className="flex items-center">
                    <i className="fas fa-memory text-secondary mr-1"></i>
                    <span>MEM-789012</span>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-border mt-4">
                <p className="text-sm text-gray-400 italic">
                  This is a placeholder interface. Backend integration with CCE is required for this feature to work.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

```

# client\src\pages\config.tsx

```tsx
import React from "react";
import { useMemoryCoreStats, useNeuralMemoryConfig, useCCEConfig } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { usePollingStore } from "@/lib/store";

// Component for displaying config key-value pairs
function ConfigItem({ label, value }: { label: string, value: string | number | boolean }) {
  return (
    <div className="py-2 border-b border-border last:border-0">
      <div className="flex justify-between items-start">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-mono text-sm bg-muted px-2 py-1 rounded max-w-[50%] break-all">
          {typeof value === "boolean" 
            ? value ? "true" : "false"
            : value.toString()}
        </span>
      </div>
    </div>
  );
}

export default function Config() {
  const { refreshAllData } = usePollingStore();
  
  // Fetch configuration data
  const memoryCoreStats = useMemoryCoreStats();
  const neuralMemoryConfig = useNeuralMemoryConfig();
  const cceConfig = useCCEConfig();
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Configuration Viewer</h2>
          <p className="text-sm text-gray-400">
            Display current runtime configurations of all services
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      <Tabs defaultValue="memory-core" className="mb-6">
        <TabsList className="mb-4">
          <TabsTrigger value="memory-core">Memory Core</TabsTrigger>
          <TabsTrigger value="neural-memory">Neural Memory</TabsTrigger>
          <TabsTrigger value="cce">Context Cascade Engine</TabsTrigger>
        </TabsList>
        
        <TabsContent value="memory-core">
          <Card>
            <CardHeader>
              <CardTitle>Memory Core Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the Memory Core service</CardDescription>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : memoryCoreStats.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Vector Index Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Index Type" value={memoryCoreStats.data.data.vector_index.index_type} />
                      <ConfigItem label="GPU Enabled" value={memoryCoreStats.data.data.vector_index.gpu_enabled} />
                      <ConfigItem label="Drift Threshold" value={100} />
                      <ConfigItem label="Dimension" value={1536} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Process Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Assembly Pruning" value={memoryCoreStats.data.data.assembly_stats.pruning_enabled} />
                      <ConfigItem label="Assembly Merging" value={memoryCoreStats.data.data.assembly_stats.merging_enabled} />
                      <ConfigItem label="Quick Recall Threshold" value={0.95} />
                      <ConfigItem label="Persistence Enabled" value={true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Memory Core configuration
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="neural-memory">
          <Card>
            <CardHeader>
              <CardTitle>Neural Memory Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the Neural Memory service</CardDescription>
            </CardHeader>
            <CardContent>
              {neuralMemoryConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : neuralMemoryConfig.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Model Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Dimensions" value={neuralMemoryConfig.data.data.dimensions} />
                      <ConfigItem label="Hidden Size" value={neuralMemoryConfig.data.data.hidden_size} />
                      <ConfigItem label="Layers" value={neuralMemoryConfig.data.data.layers} />
                      <ConfigItem label="Attention Heads" value={neuralMemoryConfig.data.data.attention_heads || 12} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Training Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Learning Rate" value={neuralMemoryConfig.data.data.learning_rate || 0.0001} />
                      <ConfigItem label="Batch Size" value={neuralMemoryConfig.data.data.batch_size || 32} />
                      <ConfigItem label="Gradient Clip" value={neuralMemoryConfig.data.data.gradient_clip || 1.0} />
                      <ConfigItem label="Emotional Boost" value={neuralMemoryConfig.data.data.emotional_boost || true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Neural Memory configuration
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="cce">
          <Card>
            <CardHeader>
              <CardTitle>Context Cascade Engine Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the CCE service</CardDescription>
            </CardHeader>
            <CardContent>
              {cceConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : cceConfig.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Variant Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Active Variant" value={cceConfig.data.data.active_variant} />
                      <ConfigItem label="Confidence Threshold" value={cceConfig.data.data.variant_confidence_threshold} />
                      <ConfigItem label="Auto Selection" value={true} />
                      <ConfigItem label="Performance Based" value={true} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">LLM Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="LLM Guidance Enabled" value={cceConfig.data.data.llm_guidance_enabled} />
                      <ConfigItem label="Retry Attempts" value={cceConfig.data.data.retry_attempts} />
                      <ConfigItem label="Timeout (ms)" value={3000} />
                      <ConfigItem label="Cache Results" value={true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load CCE configuration
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      <Card>
        <CardHeader>
          <CardTitle>Environment Variables</CardTitle>
          <CardDescription>System environment variables affecting service behavior</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-primary mb-2 uppercase">Service URLs</h3>
              <div className="space-y-0">
                <ConfigItem label="MEMORY_CORE_URL" value={process.env.MEMORY_CORE_URL || "http://memory-core:8080"} />
                <ConfigItem label="NEURAL_MEMORY_URL" value={process.env.NEURAL_MEMORY_URL || "http://neural-memory:8080"} />
                <ConfigItem label="CCE_URL" value={process.env.CCE_URL || "http://cce:8080"} />
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-primary mb-2 uppercase">Dashboard Configuration</h3>
              <div className="space-y-0">
                <ConfigItem label="NODE_ENV" value={process.env.NODE_ENV || "production"} />
                <ConfigItem label="Default Poll Rate (ms)" value={5000} />
                <ConfigItem label="Max Visible Alerts" value={10} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}

```

# client\src\pages\llm-guidance.tsx

```tsx
import React, { useState } from "react";
import { useRecentCCEResponses } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { usePollingStore } from "@/lib/store";

export default function LLMGuidance() {
  const { refreshAllData } = usePollingStore();
  const [confidenceFilter, setConfidenceFilter] = useState("all");
  const [variantFilter, setVariantFilter] = useState("all");
  const [expandedResponse, setExpandedResponse] = useState<string | null>(null);
  
  // Fetch CCE responses that include LLM guidance
  const { data, isLoading, isError } = useRecentCCEResponses();
  
  // Filter responses that have LLM advice
  const llmResponses = React.useMemo(() => {
    if (!data?.data?.recent_responses) return [];
    
    // Only include responses with LLM advice
    let filtered = data.data.recent_responses.filter(
      response => response.llm_advice_used
    );
    
    // Apply confidence filter
    if (confidenceFilter !== "all") {
      filtered = filtered.filter(response => {
        const confidence = response.llm_advice_used?.confidence_level || 0;
        
        if (confidenceFilter === "high") {
          return confidence >= 0.8;
        } else if (confidenceFilter === "medium") {
          return confidence >= 0.5 && confidence < 0.8;
        } else if (confidenceFilter === "low") {
          return confidence < 0.5;
        }
        
        return true;
      });
    }
    
    // Apply variant filter
    if (variantFilter !== "all") {
      filtered = filtered.filter(response => {
        const variantHint = response.llm_advice_used?.adjusted_advice || "";
        return variantHint.toLowerCase().includes(variantFilter.toLowerCase());
      });
    }
    
    return filtered;
  }, [data, confidenceFilter, variantFilter]);
  
  // Calculate statistics
  const stats = React.useMemo(() => {
    if (!data?.data?.recent_responses) {
      return {
        totalRequests: 0,
        avgConfidence: 0,
        variantDistribution: {}
      };
    }
    
    const llmResponses = data.data.recent_responses.filter(
      response => response.llm_advice_used
    );
    
    // Calculate average confidence
    const totalConfidence = llmResponses.reduce((acc, response) => {
      return acc + (response.llm_advice_used?.confidence_level || 0);
    }, 0);
    
    const avgConfidence = llmResponses.length > 0 
      ? totalConfidence / llmResponses.length 
      : 0;
    
    // Calculate variant distribution
    const variantDistribution: Record<string, number> = {};
    
    llmResponses.forEach(response => {
      const advice = response.llm_advice_used?.adjusted_advice || "";
      
      if (advice.toLowerCase().includes("mac-7b")) {
        variantDistribution["MAC-7b"] = (variantDistribution["MAC-7b"] || 0) + 1;
      } else if (advice.toLowerCase().includes("mac-13b")) {
        variantDistribution["MAC-13b"] = (variantDistribution["MAC-13b"] || 0) + 1;
      } else if (advice.toLowerCase().includes("titan")) {
        variantDistribution["TITAN-7b"] = (variantDistribution["TITAN-7b"] || 0) + 1;
      }
    });
    
    return {
      totalRequests: llmResponses.length,
      avgConfidence: avgConfidence,
      variantDistribution
    };
  }, [data]);
  
  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };
  
  // Get confidence badge color
  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) {
      return <Badge className="bg-green-600">High ({confidence.toFixed(2)})</Badge>;
    } else if (confidence >= 0.5) {
      return <Badge className="bg-blue-600">Medium ({confidence.toFixed(2)})</Badge>;
    } else {
      return <Badge className="bg-orange-600">Low ({confidence.toFixed(2)})</Badge>;
    }
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">LLM Guidance Monitor</h2>
          <p className="text-sm text-gray-400">
            Monitor interactions with external LLM services for context orchestration
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Total LLM Requests</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-mono">{stats.totalRequests}</p>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Average Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-mono">{stats.avgConfidence.toFixed(2)}</p>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Top Variant Hint</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-32" />
            ) : (
              <div>
                {Object.entries(stats.variantDistribution).length > 0 ? (
                  <p className="text-2xl font-mono">
                    {Object.entries(stats.variantDistribution)
                      .sort((a, b) => b[1] - a[1])[0]?.[0] || "None"}
                  </p>
                ) : (
                  <p className="text-gray-400">No data available</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      
      {/* Filters */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Filter LLM Guidance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-2">Confidence Level</p>
              <Select value={confidenceFilter} onValueChange={setConfidenceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by confidence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Confidence Levels</SelectItem>
                  <SelectItem value="high">High (≥ 0.8)</SelectItem>
                  <SelectItem value="medium">Medium (0.5 - 0.8)</SelectItem>
                  <SelectItem value="low">Low (&lt; 0.5)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Variant Hint</p>
              <Select value={variantFilter} onValueChange={setVariantFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by variant" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Variants</SelectItem>
                  <SelectItem value="mac-7b">MAC-7b</SelectItem>
                  <SelectItem value="mac-13b">MAC-13b</SelectItem>
                  <SelectItem value="titan">TITAN-7b</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* LLM Guidance Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent LLM Guidance</CardTitle>
          <CardDescription>
            {isLoading ? (
              <Skeleton className="h-4 w-48" />
            ) : (
              <>Showing {llmResponses.length} LLM guidance requests</>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-64 w-full" />
            </div>
          ) : isError ? (
            <div className="text-center py-8 text-gray-400">
              <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
              Failed to load LLM guidance data
            </div>
          ) : llmResponses.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <i className="fas fa-info-circle mr-2"></i>
              No LLM guidance data available
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[180px]">Timestamp</TableHead>
                    <TableHead>Input Summary</TableHead>
                    <TableHead>Adjusted Advice</TableHead>
                    <TableHead className="w-[120px]">Confidence</TableHead>
                    <TableHead className="w-[80px]">Raw Data</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {llmResponses.map((response, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-mono text-xs">
                        {formatTimestamp(response.timestamp)}
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xs truncate">
                          {response.variant_selection?.reason || "N/A"}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xs truncate">
                          {response.llm_advice_used?.adjusted_advice || "N/A"}
                        </div>
                      </TableCell>
                      <TableCell>
                        {getConfidenceBadge(response.llm_advice_used?.confidence_level || 0)}
                      </TableCell>
                      <TableCell>
                        <Collapsible>
                          <CollapsibleTrigger asChild>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              onClick={() => setExpandedResponse(
                                expandedResponse === response.timestamp ? null : response.timestamp
                              )}
                            >
                              <i className={`fas fa-chevron-${expandedResponse === response.timestamp ? 'up' : 'down'}`}></i>
                            </Button>
                          </CollapsibleTrigger>
                          <CollapsibleContent className="mt-2">
                            <div className="bg-muted p-3 rounded text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                              {response.llm_advice_used?.raw_advice || "Raw advice not available"}
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

```

# client\src\pages\logs.tsx

```tsx
import React, { useState, useEffect, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";

// Mock log data structure (will be replaced with WebSocket data in the future)
interface LogEntry {
  timestamp: string;
  level: "DEBUG" | "INFO" | "WARN" | "ERROR";
  service: "MemoryCore" | "NeuralMemory" | "CCE";
  message: string;
}

// Component for individual log entry
function LogEntryRow({ entry }: { entry: LogEntry }) {
  const levelColors = {
    DEBUG: "text-gray-400",
    INFO: "text-blue-400",
    WARN: "text-yellow-400",
    ERROR: "text-red-400"
  };
  
  const serviceColors = {
    MemoryCore: "border-secondary",
    NeuralMemory: "border-primary",
    CCE: "border-accent"
  };
  
  return (
    <div className={`px-3 py-2 border-l-2 ${serviceColors[entry.service]} text-xs font-mono mb-1 hover:bg-muted`}>
      <span className="text-gray-500 mr-2">{entry.timestamp}</span>
      <Badge variant="outline" className={`${levelColors[entry.level]} mr-2`}>
        {entry.level}
      </Badge>
      <Badge variant="outline" className="mr-2">
        {entry.service}
      </Badge>
      <span>{entry.message}</span>
    </div>
  );
}

export default function Logs() {
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [serviceFilter, setServiceFilter] = useState<string>("all");
  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Create placeholder text explaining this is a future feature
  const placeholderInfo = (
    <div className="text-center py-12">
      <i className="fas fa-stream text-4xl text-muted-foreground mb-4"></i>
      <h3 className="text-lg font-medium mb-2">Real-time Log Viewer</h3>
      <p className="text-muted-foreground mb-6 max-w-md mx-auto">
        This feature will connect to a WebSocket endpoint on each service to stream logs in real-time.
        It is currently a placeholder for a future implementation.
      </p>
      <div className="flex justify-center">
        <Button disabled className="mr-2">
          Connect to Log Stream
        </Button>
      </div>
    </div>
  );
  
  // Filter logs based on selected filters
  const filteredLogs = React.useMemo(() => {
    return logEntries.filter(entry => {
      // Apply service filter
      if (serviceFilter !== "all" && entry.service !== serviceFilter) {
        return false;
      }
      
      // Apply level filter
      if (levelFilter !== "all" && entry.level !== levelFilter) {
        return false;
      }
      
      // Apply search filter
      if (searchTerm && !entry.message.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      return true;
    });
  }, [logEntries, serviceFilter, levelFilter, searchTerm]);
  
  // Handle auto-scrolling
  useEffect(() => {
    if (autoScroll && scrollAreaRef.current) {
      const scrollArea = scrollAreaRef.current;
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Real-time Log Viewer</h2>
          <p className="text-sm text-gray-400">
            Stream logs from Synthians Cognitive Architecture services for real-time debugging
          </p>
        </div>
        
        <div className="flex items-center">
          <Badge 
            variant={isConnected ? "default" : "outline"} 
            className={isConnected ? "bg-green-600" : "text-gray-400"}
          >
            <div className={`w-2 h-2 rounded-full mr-1 ${isConnected ? "bg-background" : "bg-gray-400"}`}></div>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
      </div>
      
      {/* Log Controls */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <CardTitle>Log Stream Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-2">Service Filter</p>
              <Select value={serviceFilter} onValueChange={setServiceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by service" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Services</SelectItem>
                  <SelectItem value="MemoryCore">Memory Core</SelectItem>
                  <SelectItem value="NeuralMemory">Neural Memory</SelectItem>
                  <SelectItem value="CCE">CCE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Log Level</p>
              <Select value={levelFilter} onValueChange={setLevelFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="DEBUG">DEBUG</SelectItem>
                  <SelectItem value="INFO">INFO</SelectItem>
                  <SelectItem value="WARN">WARN</SelectItem>
                  <SelectItem value="ERROR">ERROR</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Search</p>
              <Input
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                id="auto-scroll"
                checked={autoScroll}
                onCheckedChange={setAutoScroll}
              />
              <label htmlFor="auto-scroll" className="text-sm">Auto-scroll</label>
            </div>
            
            <div>
              <Button disabled={isConnected} variant="outline" className="mr-2">
                Connect
              </Button>
              <Button disabled={!isConnected} variant="outline">
                Clear Logs
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Log Viewer */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <CardTitle>Log Stream</CardTitle>
            <Badge variant="outline">
              {filteredLogs.length} entries
            </Badge>
          </div>
          <CardDescription>
            Streaming logs will appear here once connected
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-card border border-border rounded-md h-[500px]">
            {placeholderInfo}
            
            {/* Log entries would go here in a ScrollArea once implemented */}
            {/* <ScrollArea className="h-[500px] p-2" ref={scrollAreaRef}>
              {filteredLogs.map((entry, index) => (
                <LogEntryRow key={index} entry={entry} />
              ))}
              {filteredLogs.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-info-circle mr-2"></i>
                  No log entries match your filters
                </div>
              )}
            </ScrollArea> */}
          </div>
        </CardContent>
      </Card>
    </>
  );
}

```

# client\src\pages\memory-core.tsx

```tsx
import React, { useState } from "react";
import { useMemoryCoreHealth, useMemoryCoreStats, useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { usePollingStore } from "@/lib/store";

export default function MemoryCore() {
  const { refreshAllData } = usePollingStore();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Fetch Memory Core data
  const memoryCoreHealth = useMemoryCoreHealth();
  const memoryCoreStats = useMemoryCoreStats();
  const assemblies = useAssemblies();
  
  // Prepare service status object
  const serviceStatus = memoryCoreHealth.data?.data ? {
    name: "Memory Core",
    status: memoryCoreHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || "Unknown",
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  // Calculate warning thresholds
  const isDriftAboveWarning = memoryCoreStats.data?.data?.vector_index?.drift_count > 50;
  const isDriftAboveCritical = memoryCoreStats.data?.data?.vector_index?.drift_count > 100;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Memory Core Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">SynthiansMemoryCore</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {serviceStatus ? (
              <ServiceStatus service={serviceStatus} />
            ) : (
              <Skeleton className="h-5 w-20" />
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
            <div>
              <p className="text-sm text-gray-500 mb-1">Connection</p>
              {memoryCoreHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus ? (
                <p className="text-lg">{serviceStatus.url}</p>
              ) : (
                <p className="text-red-500">Unreachable</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Uptime</p>
              {memoryCoreHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus?.uptime ? (
                <p className="text-lg">{serviceStatus.uptime}</p>
              ) : (
                <p className="text-gray-400">Unknown</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Version</p>
              {memoryCoreHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus?.version ? (
                <p className="text-lg">{serviceStatus.version}</p>
              ) : (
                <p className="text-gray-400">Unknown</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Tabs for different views */}
      <Tabs defaultValue="overview" className="mb-6">
        <TabsList>
          <TabsTrigger value="overview" onClick={() => setActiveTab("overview")}>Overview</TabsTrigger>
          <TabsTrigger value="vector-index" onClick={() => setActiveTab("vector-index")}>Vector Index</TabsTrigger>
          <TabsTrigger value="assemblies" onClick={() => setActiveTab("assemblies")}>Assemblies</TabsTrigger>
          <TabsTrigger value="persistence" onClick={() => setActiveTab("persistence")}>Persistence</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="col-span-2">
              <CardHeader>
                <CardTitle>Core Stats</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryCoreStats.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ) : memoryCoreStats.data?.data ? (
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Memories</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.total_memories.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Dirty Items</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.dirty_items.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Assemblies</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.total_assemblies.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Pending Vector Updates</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.pending_vector_updates.toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400">Failed to load Memory Core stats</p>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Performance</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryCoreStats.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-8 w-full" />
                  </div>
                ) : memoryCoreStats.data?.data?.performance ? (
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <p className="text-sm text-gray-500">Quick Recall Rate</p>
                        <p className="text-sm font-mono">
                          {memoryCoreStats.data.data.performance.quick_recall_rate.toFixed(2)}%
                        </p>
                      </div>
                      <Progress value={memoryCoreStats.data.data.performance.quick_recall_rate} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <p className="text-sm text-gray-500">Threshold Recall Rate</p>
                        <p className="text-sm font-mono">
                          {memoryCoreStats.data.data.performance.threshold_recall_rate.toFixed(2)}%
                        </p>
                      </div>
                      <Progress value={memoryCoreStats.data.data.performance.threshold_recall_rate} className="h-2" />
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400">Performance data unavailable</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="vector-index" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Vector Index Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.data?.data?.vector_index ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Vector Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.vector_index.count.toLocaleString()}
                      </p>
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Mapping Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.vector_index.mapping_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Drift Count</p>
                      <div className="flex items-center">
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.vector_index.drift_count.toLocaleString()}
                        </p>
                        {isDriftAboveCritical && (
                          <Badge variant="destructive" className="ml-2">Critical</Badge>
                        )}
                        {isDriftAboveWarning && !isDriftAboveCritical && (
                          <Badge variant="outline" className="ml-2 text-yellow-400 border-yellow-400">Warning</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Index Type</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.vector_index.index_type}
                      </p>
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">GPU Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.vector_index.gpu_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Vector index data unavailable</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="assemblies" className="mt-4">
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Assembly Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.data?.data?.assembly_stats ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Total Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.total_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Average Size</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.average_size.toFixed(1)}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Indexed Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.indexed_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Pruning Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.assembly_stats.pruning_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Vector Indexed Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.vector_indexed_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Merging Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.assembly_stats.merging_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Assembly stats unavailable</p>
              )}
            </CardContent>
          </Card>
          
          <AssemblyTable
            assemblies={assemblies.data?.data || null}
            isLoading={assemblies.isLoading}
            title="All Assemblies"
          />
        </TabsContent>
        
        <TabsContent value="persistence" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Persistence Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.data?.data?.persistence ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-500 mb-2">Last Update</p>
                    <p className="text-lg">
                      {new Date(memoryCoreStats.data.data.persistence.last_update).toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 mb-2">Last Backup</p>
                    <p className="text-lg">
                      {new Date(memoryCoreStats.data.data.persistence.last_backup).toLocaleString()}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Persistence data unavailable</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}

```

# client\src\pages\memory-core\diagnostics.tsx

```tsx
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useMergeLog, useRuntimeConfig } from '@/lib/api';
import { MergeLogView } from '@/components/dashboard/MergeLogView';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { RefreshCw } from 'lucide-react';
import { MergeLogEntry } from '@shared/schema';

export default function MemoryCoreDiagnostics() {
  const [selectedService, setSelectedService] = useState<string>('memory-core');
  const [logLimit, setLogLimit] = useState<number>(50);
  
  // Fetch merge log data
  const mergeLogQuery = useMergeLog(logLimit);
  
  // Fetch runtime configuration
  const configQuery = useRuntimeConfig(selectedService);
  
  // Handle refresh for both queries
  const handleRefresh = () => {
    mergeLogQuery.refetch();
    configQuery.refetch();
  };
  
  return (
    <div className="container py-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold">Memory Core Diagnostics</h1>
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
        
        {/* Merge Log */}
        <MergeLogView
          entries={mergeLogQuery.data as MergeLogEntry[] | undefined}
          isLoading={mergeLogQuery.isLoading}
          isError={mergeLogQuery.isError}
          error={mergeLogQuery.error as Error}
        />
        
        {/* Runtime Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Runtime Configuration</CardTitle>
            <CardDescription>
              View current runtime configuration values for the selected service.
            </CardDescription>
            <div className="flex items-center gap-2 mt-2">
              <Select value={selectedService} onValueChange={setSelectedService}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select service" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="memory-core">Memory Core</SelectItem>
                  <SelectItem value="neural-memory">Neural Memory</SelectItem>
                  <SelectItem value="cce">Controlled Context Exchange</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {configQuery.isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
              </div>
            ) : configQuery.isError ? (
              <div className="p-4 text-center">
                <p className="text-red-500">
                  {configQuery.error?.message || 'Failed to load configuration data'}
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b text-xs text-muted-foreground">
                      <th className="text-left py-2 font-medium">Parameter</th>
                      <th className="text-left py-2 font-medium">Value</th>
                      <th className="text-left py-2 font-medium">Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {Object.entries(configQuery.data || {}).map(([key, value]) => (
                      <tr key={key} className="hover:bg-muted/50">
                        <td className="py-2 font-mono text-sm">{key}</td>
                        <td className="py-2 font-mono text-sm">
                          {typeof value === 'object' 
                            ? JSON.stringify(value)
                            : String(value)}
                        </td>
                        <td className="py-2 text-sm text-muted-foreground">
                          {Array.isArray(value) 
                            ? 'array' 
                            : typeof value === 'object' && value !== null 
                              ? 'object' 
                              : typeof value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

```

# client\src\pages\neural-memory.tsx

```tsx
import React, { useState } from "react";
import { useNeuralMemoryHealth, useNeuralMemoryStatus, useNeuralMemoryDiagnostics } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { usePollingStore } from "@/lib/store";

export default function NeuralMemory() {
  const { refreshAllData } = usePollingStore();
  const [timeWindow, setTimeWindow] = useState("12h");
  
  // Fetch Neural Memory data
  const neuralMemoryHealth = useNeuralMemoryHealth();
  const neuralMemoryStatus = useNeuralMemoryStatus();
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeWindow);
  
  // Prepare service status object
  const serviceStatus = neuralMemoryHealth.data?.data ? {
    name: "Neural Memory",
    status: neuralMemoryHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/neural-memory/health",
    uptime: neuralMemoryHealth.data.data.uptime || "Unknown",
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } : null;
  
  // Prepare chart data
  const prepareChartData = () => {
    if (!neuralMemoryDiagnostics.data?.data?.history) {
      return [];
    }
    
    return neuralMemoryDiagnostics.data.data.history.map((item: any) => ({
      timestamp: item.timestamp,
      loss: item.loss,
      grad_norm: item.grad_norm,
      qr_boost: item.qr_boost
    }));
  };
  
  const chartData = prepareChartData();
  
  // Determine if any metrics are in warning/critical state
  const isGradNormHigh = 
    neuralMemoryDiagnostics.data?.data?.avg_grad_norm > 0.8;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Neural Memory Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">NeuralMemoryModule</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {serviceStatus ? (
              <ServiceStatus service={serviceStatus} />
            ) : (
              <Skeleton className="h-5 w-20" />
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
            <div>
              <p className="text-sm text-gray-500 mb-1">Connection</p>
              {neuralMemoryHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus ? (
                <p className="text-lg">{serviceStatus.url}</p>
              ) : (
                <p className="text-red-500">Unreachable</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Uptime</p>
              {neuralMemoryHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus?.uptime ? (
                <p className="text-lg">{serviceStatus.uptime}</p>
              ) : (
                <p className="text-gray-400">Unknown</p>
              )}
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Version</p>
              {neuralMemoryHealth.isLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : serviceStatus?.version ? (
                <p className="text-lg">{serviceStatus.version}</p>
              ) : (
                <p className="text-gray-400">Unknown</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Configuration Overview */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Neural Memory Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          {neuralMemoryStatus.isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : neuralMemoryStatus.data?.data ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Initialization Status</p>
                <p className="text-lg">
                  {neuralMemoryStatus.data.data.initialized ? (
                    <span className="text-green-400">Initialized</span>
                  ) : (
                    <span className="text-yellow-400">Not Initialized</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Dimensions</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.data.config?.dimensions || "Unknown"}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Hidden Size</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.data.config?.hidden_size || "Unknown"}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-gray-400">Failed to load Neural Memory status</p>
          )}
        </CardContent>
      </Card>
      
      {/* Warning if high grad norm */}
      {isGradNormHigh && neuralMemoryDiagnostics.data?.data && (
        <Alert variant="destructive" className="mb-6">
          <AlertTitle className="flex items-center">
            <i className="fas fa-exclamation-circle mr-2"></i>
            High Gradient Norm Detected
          </AlertTitle>
          <AlertDescription>
            The gradient norm of {neuralMemoryDiagnostics.data.data.avg_grad_norm.toFixed(4)} exceeds the recommended threshold of 0.7500.
          </AlertDescription>
        </Alert>
      )}
      
      {/* Tabs for Performance Metrics */}
      <Tabs defaultValue="performance" className="mb-6">
        <TabsList>
          <TabsTrigger value="performance">Performance Metrics</TabsTrigger>
          <TabsTrigger value="emotional">Emotional Loop</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>
        
        <TabsContent value="performance" className="mt-4">
          <div className="grid grid-cols-1 gap-6">
            <MetricsChart
              title="Neural Memory Performance"
              data={chartData}
              dataKeys={[
                { key: "loss", color: "#FF008C", name: "Loss" },
                { key: "grad_norm", color: "#1EE4FF", name: "Gradient Norm" },
                { key: "qr_boost", color: "#FF3EE8", name: "QR Boost" }
              ]}
              isLoading={neuralMemoryDiagnostics.isLoading}
              timeRange={timeWindow}
              onTimeRangeChange={setTimeWindow}
              summary={[
                { 
                  label: "Avg. Loss", 
                  value: neuralMemoryDiagnostics.data?.data?.avg_loss.toFixed(4) || "--", 
                  color: "text-primary" 
                },
                { 
                  label: "Avg. Grad Norm", 
                  value: neuralMemoryDiagnostics.data?.data?.avg_grad_norm.toFixed(4) || "--",
                  color: isGradNormHigh ? "text-destructive" : "text-secondary"
                },
                { 
                  label: "Avg. QR Boost", 
                  value: neuralMemoryDiagnostics.data?.data?.avg_qr_boost.toFixed(4) || "--", 
                  color: "text-accent" 
                }
              ]}
            />
          </div>
        </TabsContent>
        
        <TabsContent value="emotional" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Emotional Loop Diagnostics</CardTitle>
            </CardHeader>
            <CardContent>
              {neuralMemoryDiagnostics.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : neuralMemoryDiagnostics.data?.data?.emotional_loop ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500 mb-1">Dominant Emotions</p>
                      <div className="flex flex-wrap gap-2">
                        {neuralMemoryDiagnostics.data.data.emotional_loop.dominant_emotions.map((emotion: string, idx: number) => (
                          <Badge key={idx} variant="outline" className="text-primary border-primary">
                            {emotion}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Entropy</p>
                      <p className="text-lg font-mono">
                        {neuralMemoryDiagnostics.data.data.emotional_loop.entropy.toFixed(4)}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500 mb-1">Bias Index</p>
                      <div>
                        <p className="text-lg font-mono mb-1">
                          {neuralMemoryDiagnostics.data.data.emotional_loop.bias_index.toFixed(4)}
                        </p>
                        <Progress 
                          value={neuralMemoryDiagnostics.data.data.emotional_loop.bias_index * 100} 
                          className="h-2"
                        />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Match Rate</p>
                      <div>
                        <p className="text-lg font-mono mb-1">
                          {(neuralMemoryDiagnostics.data.data.emotional_loop.match_rate * 100).toFixed(2)}%
                        </p>
                        <Progress 
                          value={neuralMemoryDiagnostics.data.data.emotional_loop.match_rate * 100} 
                          className="h-2"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Emotional loop diagnostics unavailable</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="recommendations" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                {neuralMemoryDiagnostics.isLoading ? (
                  <div className="space-y-2">
                    <Skeleton className="h-6 w-full" />
                    <Skeleton className="h-6 w-full" />
                    <Skeleton className="h-6 w-full" />
                  </div>
                ) : neuralMemoryDiagnostics.data?.data?.alerts ? (
                  neuralMemoryDiagnostics.data.data.alerts.length > 0 ? (
                    <ul className="space-y-2">
                      {neuralMemoryDiagnostics.data.data.alerts.map((alert: string, idx: number) => (
                        <li key={idx} className="flex items-start">
                          <i className="fas fa-exclamation-triangle text-yellow-400 mr-2 mt-1"></i>
                          <span>{alert}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-400">No alerts detected</p>
                  )
                ) : (
                  <p className="text-gray-400">Alert data unavailable</p>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                {neuralMemoryDiagnostics.isLoading ? (
                  <div className="space-y-2">
                    <Skeleton className="h-6 w-full" />
                    <Skeleton className="h-6 w-full" />
                    <Skeleton className="h-6 w-full" />
                  </div>
                ) : neuralMemoryDiagnostics.data?.data?.recommendations ? (
                  neuralMemoryDiagnostics.data.data.recommendations.length > 0 ? (
                    <ul className="space-y-2">
                      {neuralMemoryDiagnostics.data.data.recommendations.map((rec: string, idx: number) => (
                        <li key={idx} className="flex items-start">
                          <i className="fas fa-lightbulb text-secondary mr-2 mt-1"></i>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-400">No recommendations available</p>
                  )
                ) : (
                  <p className="text-gray-400">Recommendation data unavailable</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </>
  );
}

```

# client\src\pages\not-found.tsx

```tsx
import { Card, CardContent } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-md mx-4">
        <CardContent className="pt-6">
          <div className="flex mb-4 gap-2">
            <AlertCircle className="h-8 w-8 text-red-500" />
            <h1 className="text-2xl font-bold text-gray-900">404 Page Not Found</h1>
          </div>

          <p className="mt-4 text-sm text-gray-600">
            Did you forget to add the page to the router?
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

```

# client\src\pages\overview.tsx

```tsx
import React, { useState } from "react";
import { OverviewCard } from "@/components/dashboard/OverviewCard";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { SystemArchitecture } from "@/components/dashboard/SystemArchitecture";
import { DiagnosticAlerts } from "@/components/dashboard/DiagnosticAlerts";
import { CCEChart } from "@/components/dashboard/CCEChart";
import { 
  useMemoryCoreHealth,
  useNeuralMemoryHealth,
  useCCEHealth,
  useMemoryCoreStats,
  useAssemblies,
  useNeuralMemoryDiagnostics,
  useRecentCCEResponses,
  useAlerts
} from "@/lib/api";

export default function Overview() {
  const [timeRange, setTimeRange] = useState<string>("12h");
  
  // Fetch all the required data
  const memoryCoreHealth = useMemoryCoreHealth();
  const neuralMemoryHealth = useNeuralMemoryHealth();
  const cceHealth = useCCEHealth();
  const memoryCoreStats = useMemoryCoreStats();
  const assemblies = useAssemblies();
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeRange);
  const recentCCEResponses = useRecentCCEResponses();
  const alerts = useAlerts();
  
  // Prepare data for Memory Core status card
  const memoryCoreService = memoryCoreHealth.data?.data ? {
    name: "Memory Core",
    status: memoryCoreHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || "Unknown",
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  const memoryCoreMetrics = memoryCoreStats.data?.data ? {
    "Total Memories": memoryCoreStats.data.data.total_memories.toLocaleString(),
    "Total Assemblies": memoryCoreStats.data.data.total_assemblies.toLocaleString()
  } : null;
  
  // Prepare data for Neural Memory status card
  const neuralMemoryService = neuralMemoryHealth.data?.data ? {
    name: "Neural Memory",
    status: neuralMemoryHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/neural-memory/health",
    uptime: neuralMemoryHealth.data.data.uptime || "Unknown",
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } : null;
  
  const neuralMemoryMetrics = neuralMemoryDiagnostics.data?.data ? {
    "Avg. Loss": neuralMemoryDiagnostics.data.data.avg_loss.toFixed(4),
    "Grad Norm": neuralMemoryDiagnostics.data.data.avg_grad_norm.toFixed(4)
  } : null;
  
  // Prepare data for CCE status card
  const cceService = cceHealth.data?.data ? {
    name: "Context Cascade Engine",
    status: cceHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || "Unknown",
    version: cceHealth.data.data.version || "Unknown"
  } : null;
  
  const cceMetrics = recentCCEResponses.data?.data?.recent_responses ? {
    "Active Titan Variant": recentCCEResponses.data.data.recent_responses[0]?.variant_output?.variant_type || "Unknown"
  } : null;
  
  // Prepare data for Neural Memory chart
  const prepareNeuralMemoryChartData = () => {
    const emptyData = Array(12).fill(0).map((_, i) => ({
      timestamp: new Date(Date.now() - i * 3600 * 1000).toISOString(),
      loss: Math.random() * 0.05 + 0.02, // Placeholder values when no real data
      grad_norm: Math.random() * 0.2 + 0.7
    }));
    
    if (!neuralMemoryDiagnostics.data?.data?.history) {
      return emptyData;
    }
    
    return neuralMemoryDiagnostics.data.data.history.map((item: any) => ({
      timestamp: item.timestamp,
      loss: item.loss,
      grad_norm: item.grad_norm
    }));
  };
  
  const neuralMemoryChartData = prepareNeuralMemoryChartData();
  
  // Prepare assemblies data
  const recentAssemblies = assemblies.data?.data?.slice(0, 5) || null;
  
  return (
    <>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white mb-1">System Overview</h2>
        <p className="text-sm text-gray-400">At-a-glance status of all core services</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <OverviewCard
          title="Memory Core"
          icon="database"
          service={memoryCoreService}
          metrics={memoryCoreMetrics}
          isLoading={memoryCoreHealth.isLoading || memoryCoreStats.isLoading}
        />
        
        <OverviewCard
          title="Neural Memory"
          icon="brain"
          service={neuralMemoryService}
          metrics={neuralMemoryMetrics}
          isLoading={neuralMemoryHealth.isLoading || neuralMemoryDiagnostics.isLoading}
        />
        
        <OverviewCard
          title="Context Cascade Engine"
          icon="sitemap"
          service={cceService}
          metrics={cceMetrics}
          isLoading={cceHealth.isLoading || recentCCEResponses.isLoading}
        />
      </div>

      {/* Performance Metrics */}
      <div className="mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <MetricsChart
            title="Neural Memory - Training Loss"
            data={neuralMemoryChartData}
            dataKeys={[
              { key: "loss", color: "#FF008C", name: "Avg. Loss" },
              { key: "grad_norm", color: "#1EE4FF", name: "Grad Norm" }
            ]}
            isLoading={neuralMemoryDiagnostics.isLoading}
            timeRange={timeRange}
            onTimeRangeChange={setTimeRange}
            summary={[
              { label: "Current", value: neuralMemoryDiagnostics.data?.data?.avg_loss.toFixed(4) || "--", color: "text-primary" },
              { label: "Min (12h)", value: "0.0341", color: "text-secondary" },
              { label: "Max (12h)", value: "0.0729", color: "text-yellow-400" }
            ]}
          />
          
          <CCEChart
            title="CCE - Variant Selection"
            data={recentCCEResponses.data?.data?.recent_responses || []}
            isLoading={recentCCEResponses.isLoading}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="mb-8">
        <AssemblyTable
          assemblies={recentAssemblies}
          isLoading={assemblies.isLoading}
          title="Last Updated Assemblies"
        />
      </div>

      {/* System Architecture */}
      <div className="mb-8">
        <SystemArchitecture />
      </div>

      {/* Diagnostic Alerts */}
      <DiagnosticAlerts
        alerts={alerts.data?.data || null}
        isLoading={alerts.isLoading}
      />
    </>
  );
}

```

# client\vite-env.d.ts

```ts
/// <reference types="vite/client" />

```

# CONTRIBUTING.md

```md
# Contributing to Synthians Cognitive Dashboard

Thank you for considering contributing to the Synthians Cognitive Dashboard! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand expected behavior.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it hasn't, create a new issue with a clear description, including:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- Environment details (browser, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include mockups or examples if possible

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Install dependencies:
\`\`\`bash
npm install
\`\`\`

2. Start the development server:
\`\`\`bash
npm run dev
\`\`\`

## Coding Standards

### TypeScript

- Use TypeScript for all new code
- Ensure proper typing for all variables, parameters, and return values
- Follow the existing project structure for new files

### React Components

- Use functional components with hooks
- Keep components small and focused on a single responsibility
- Use the shadcn component library for consistent UI

### Styling

- Use TailwindCSS for styling
- Follow the existing theme configuration
- Use the provided color variables for consistency

### Testing

- Write tests for new features using the testing framework in place
- Ensure all tests pass before submitting a PR
- Aim for good test coverage for new code

## Commit Guidelines

- Use clear, concise commit messages
- Reference issue numbers in commit messages when applicable
- Keep commits focused on a single logical change

## Documentation

- Update documentation when changing functionality
- Document new features, including:
  - Usage examples
  - API documentation
  - Configuration options

## Release Process

1. Version bump follows semantic versioning (MAJOR.MINOR.PATCH)
2. Releases are created from the main branch
3. Release notes document all significant changes

Thank you for contributing to the Synthians Cognitive Dashboard!
```

# dev.ps1

```ps1
# Set environment variables for development
$env:NODE_ENV = "development"

# Run the server with the proper import flag
node --import tsx/esm server/index.ts

```

# docs\API_REFRENCE.md

```md
# Synthians Cognitive Dashboard - API Reference

This document provides details on the API endpoints used by the Synthians Cognitive Dashboard to interact with the underlying Cognitive Architecture services.

## Base URLs

The dashboard interacts with three primary services:

- **Memory Core**: `http://memory-core:8080` (configurable via `MEMORY_CORE_URL` env variable)
- **Neural Memory**: `http://neural-memory:8080` (configurable via `NEURAL_MEMORY_URL` env variable)
- **CCE (Controlled Context Exchange)**: `http://cce:8080` (configurable via `CCE_URL` env variable)

## Authentication

Currently, the API endpoints do not require authentication. This will be implemented in future versions.

## Response Format

All API responses follow a standard format:

\`\`\`json
{
  "status": "success" | "error",
  "data": {
    // Response data specific to the endpoint
  },
  "message": "Optional message, typically for errors"
}
\`\`\`

## Common Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Resource Not Found
- `500` - Internal Server Error

## Memory Core Endpoints

### Health Check

\`\`\`
GET /api/memory-core/health
\`\`\`

Returns the health status of the Memory Core service.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "name": "Memory Core",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://memory-core:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
\`\`\`

### Memory Stats

\`\`\`
GET /api/memory-core/stats
\`\`\`

Returns statistics about the memory storage.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "total_memories": 12500,
    "total_assemblies": 450,
    "dirty_items": 12,
    "pending_vector_updates": 3,
    "vector_index": {
      "count": 12500,
      "mapping_count": 12500,
      "drift_count": 2,
      "index_type": "HNSW",
      "gpu_enabled": true
    },
    "assembly_stats": {
      "total_count": 450,
      "indexed_count": 450,
      "vector_indexed_count": 448,
      "average_size": 27.8,
      "pruning_enabled": true,
      "merging_enabled": true
    },
    "persistence": {
      "last_update": "2025-03-15T14:32:11Z",
      "last_backup": "2025-03-15T12:00:00Z"
    },
    "performance": {
      "quick_recall_rate": 0.954,
      "threshold_recall_rate": 0.892
    }
  }
}
\`\`\`

### List Assemblies

\`\`\`
GET /api/memory-core/assemblies
\`\`\`

Returns a list of all memory assemblies.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": [
    {
      "id": "assembly-123",
      "name": "Core Concepts",
      "description": "Fundamental AI concepts",
      "member_count": 145,
      "keywords": ["AI", "concepts", "foundation"],
      "tags": ["important", "core"],
      "topics": ["learning", "reasoning"],
      "created_at": "2025-01-15T08:12:34Z",
      "updated_at": "2025-03-14T16:45:22Z",
      "vector_index_updated_at": "2025-03-14T16:46:01Z",
      "memory_ids": ["mem-123", "mem-124", "mem-125"]
    },
    // More assemblies...
  ]
}
\`\`\`

### Get Assembly Details

\`\`\`
GET /api/memory-core/assemblies/:id
\`\`\`

Returns details about a specific assembly.

**Parameters**:
- `id` (path parameter): The ID of the assembly

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "id": "assembly-123",
    "name": "Core Concepts",
    "description": "Fundamental AI concepts",
    "member_count": 145,
    "keywords": ["AI", "concepts", "foundation"],
    "tags": ["important", "core"],
    "topics": ["learning", "reasoning"],
    "created_at": "2025-01-15T08:12:34Z",
    "updated_at": "2025-03-14T16:45:22Z",
    "vector_index_updated_at": "2025-03-14T16:46:01Z",
    "memory_ids": ["mem-123", "mem-124", "mem-125"],
    "memories": [
      {
        "id": "mem-123",
        "content": "Understanding of basic neural networks",
        "created_at": "2025-01-15T08:12:34Z",
        "type": "concept"
      },
      // More memories...
    ]
  }
}
\`\`\`

### Verify Vector Index

\`\`\`
POST /api/memory-core/verify-index
\`\`\`

Triggers a verification of the vector index.

**Response**:
\`\`\`json
{
  "status": "success",
  "message": "Vector index verification initiated",
  "data": {
    "job_id": "verify-job-456",
    "estimated_completion_time": "2025-03-16T15:30:00Z"
  }
}
\`\`\`

### Trigger Retry Loop

\`\`\`
POST /api/memory-core/retry-loop
\`\`\`

Triggers the retry loop for failed operations.

**Response**:
\`\`\`json
{
  "status": "success",
  "message": "Retry loop triggered",
  "data": {
    "pending_operations": 3
  }
}
\`\`\`

## Neural Memory Endpoints

### Health Check

\`\`\`
GET /api/neural-memory/health
\`\`\`

Returns the health status of the Neural Memory service.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "name": "Neural Memory",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://neural-memory:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
\`\`\`

### Neural Memory Status

\`\`\`
GET /api/neural-memory/status
\`\`\`

Returns the status of the Neural Memory system.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "initialized": true,
    "config": {
      "dimensions": 1536,
      "hidden_size": 768,
      "layers": 12
    }
  }
}
\`\`\`

### Emotional Loop Diagnostics

\`\`\`
GET /api/neural-memory/diagnose_emoloop
\`\`\`

Returns diagnostic information about the emotional loop.

**Parameters**:
- `window` (query parameter): Time window for diagnostics (default: "24h")

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "avg_loss": 0.0324,
    "avg_grad_norm": 0.0512,
    "avg_qr_boost": 0.1786,
    "emotional_loop": {
      "dominant_emotions": ["curiosity", "confidence"],
      "entropy": 0.7821,
      "bias_index": 0.1232,
      "match_rate": 0.8934
    },
    "alerts": [
      "Gradient instability detected at 14:23:11"
    ],
    "recommendations": [
      "Consider reducing learning rate to stabilize training"
    ]
  }
}
\`\`\`

### Initialize Neural Memory

\`\`\`
POST /api/neural-memory/initialize
\`\`\`

Initializes or resets the Neural Memory system.

**Response**:
\`\`\`json
{
  "status": "success",
  "message": "Neural Memory initialized successfully",
  "data": {
    "initialization_time": "2025-03-16T14:23:11Z",
    "config": {
      "dimensions": 1536,
      "hidden_size": 768,
      "layers": 12
    }
  }
}
\`\`\`

## CCE (Controlled Context Exchange) Endpoints

### Health Check

\`\`\`
GET /api/cce/health
\`\`\`

Returns the health status of the CCE service.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "name": "Context Cascade Engine",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://cce:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
\`\`\`

### CCE Status

\`\`\`
GET /api/cce/status
\`\`\`

Returns the status of the CCE system.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "active_variant": "MAC-13b",
    "llm_guidance_enabled": true,
    "recent_success_rate": 0.978,
    "average_latency": 234.5
  }
}
\`\`\`

### Recent CCE Responses

\`\`\`
GET /api/cce/metrics/recent_cce_responses
\`\`\`

Returns recent CCE responses with metrics.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": {
    "recent_responses": [
      {
        "timestamp": "2025-03-16T14:23:11Z",
        "status": "success",
        "variant_output": {
          "variant_type": "MAC-13b"
        },
        "variant_selection": {
          "selected_variant": "MAC-13b",
          "reason": "High precision required for technical context",
          "performance_used": true
        },
        "llm_advice_used": {
          "raw_advice": "Consider using MAC-13b for this technical query",
          "adjusted_advice": "Using MAC-13b for optimal technical reasoning",
          "confidence_level": 0.89,
          "adjustment_reason": "Enhanced with system parameters"
        }
      },
      // More responses...
    ]
  }
}
\`\`\`

### Set CCE Variant

\`\`\`
POST /api/cce/set-variant
\`\`\`

Sets the active variant for the CCE.

**Request Body**:
\`\`\`json
{
  "variant": "MAC-13b" | "MAC-7b" | "TITAN-7b"
}
\`\`\`

**Response**:
\`\`\`json
{
  "status": "success",
  "message": "Variant set successfully",
  "data": {
    "previous_variant": "MAC-7b",
    "new_variant": "MAC-13b",
    "change_timestamp": "2025-03-16T14:25:11Z"
  }
}
\`\`\`

## System-wide Endpoints

### Alerts

\`\`\`
GET /api/alerts
\`\`\`

Returns system-wide alerts from all services.

**Response**:
\`\`\`json
{
  "status": "success",
  "data": [
    {
      "id": "alert-1",
      "type": "error" | "warning" | "info",
      "title": "High gradient detected",
      "description": "Neural Memory training shows unusually high gradients",
      "timestamp": "2025-03-16T13:24:56Z",
      "source": "NeuralMemory",
      "action": "Consider pausing training"
    },
    // More alerts...
  ]
}
\`\`\`

## Error Responses

When an error occurs, the API will return a response with an error status:

\`\`\`json
{
  "status": "error",
  "message": "Detailed error message",
  "code": "ERROR_CODE"
}
\`\`\`

Common error codes include:

- `SERVICE_UNAVAILABLE` - The service is not accessible
- `INVALID_PARAMETERS` - The request contains invalid parameters
- `RESOURCE_NOT_FOUND` - The requested resource does not exist
- `INTERNAL_ERROR` - An unexpected error occurred in the service

## Rate Limiting

Currently, there are no rate limits on the API endpoints. This may change in future versions.

## Versioning

The current API version is v1. The version is not included in the URL path as there is only one version currently.

## Future Endpoints

The following endpoints are planned for future releases:

- Streaming log endpoints via WebSockets
- Authentication endpoints
- User management endpoints
- Detailed memory search endpoints
- Batch operations for assemblies and memories
```

# docs\ARCHITECHTURE.md

```md
# Synthians Cognitive Dashboard - Architecture

This document outlines the architecture of the Synthians Cognitive Dashboard, describing the key components, data flow, and design decisions.

## System Overview

The Synthians Cognitive Dashboard follows a standard client-server pattern, but with a specific twist: the "server" component acts primarily as a **Backend-For-Frontend (BFF) proxy**.

## Architecture Diagram

\`\`\`mermaid
graph LR
    subgraph Browser
        A[React Frontend App]
    end

    subgraph Dashboard Server (Node.js/Express)
        B(Backend Proxy Server)
        B -- Serves --> A
        B -- API Proxy --> C[Memory Core API]
        B -- API Proxy --> D[Neural Memory API]
        B -- API Proxy --> E[CCE API]
    end

    subgraph Synthians Core Services
        C(Memory Core Service)
        D(Neural Memory Service)
        E(Context Cascade Engine)
    end

    A -- HTTP Request /api/... --> B


The Synthians Cognitive Dashboard is a web-based monitoring and management interface for the Synthians AI system. It provides real-time visibility and control for the three core services that make up the Synthians Cognitive Architecture:

1. **Memory Core** - Manages the storage and retrieval of episodic and semantic memories
2. **Neural Memory** - Handles vector embedding generation and memory association
3. **Controlled Context Exchange (CCE)** - Orchestrates information flow between components

The dashboard follows a client-server architecture, with a React frontend and an Express.js backend that proxies requests to the underlying services.

## Architecture Diagram

\`\`\`
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Memory Core   │         │  Neural Memory  │         │       CCE       │
│    Service      │         │    Service      │         │    Service      │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         │                           │                           │
         └───────────────────┬───────────────────┬──────────────┘
                             │                   │
                     ┌───────┴───────┐   ┌───────┴───────┐
                     │               │   │               │
                     │ Express.js    │◄──┤ WebSocket     │
                     │ Backend       │   │ Server        │
                     │               │   │               │
                     └───────┬───────┘   └───────────────┘
                             │                   
                     ┌───────┴───────┐           
                     │               │           
                     │  React        │           
                     │  Frontend     │           
                     │               │           
                     └───────────────┘           
\`\`\`

## Components

### Frontend

The frontend is a React application built with TypeScript and organized into the following main directories:

- **/components** - Reusable UI components
  - **/dashboard** - Dashboard-specific components like metric charts
  - **/layout** - Layout components like sidebar and topbar
  - **/ui** - Generic UI components (built on shadcn/ui)
- **/hooks** - Custom React hooks
- **/lib** - Utilities and API clients
- **/pages** - Page components corresponding to routes

Key frontend technologies include:

- **React** - Component-based UI library
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Utility-first CSS framework
- **Shadcn UI** - Component primitives
- **TanStack Query** - Data fetching and caching
- **Zustand** - State management
- **Recharts** - Data visualization
- **Wouter** - Routing

#### Component Structure

The dashboard follows a hierarchical component structure:

1. **App.tsx** - The root component that sets up routing and providers
2. **DashboardShell** - Provides the application layout with sidebar and topbar
   - **Sidebar** - Navigation menu with links to different sections
   - **TopBar** - Header with search, refresh controls, and status indicators
3. **Page Components** - Main content areas for each route
4. **UI Components** - Reusable elements like buttons, cards, and toasts

#### JSX Runtime and React Imports

The application uses React 18's automatic JSX runtime transformation, but explicit React imports are still required in all components that use React features like hooks or JSX. The Vite configuration is set up to handle path aliases and proper JSX transformation.

### Backend

The backend is an Express.js application that serves the React frontend and provides API routes. The server has several key responsibilities:

1. **Proxy API Requests** - Forward requests to the underlying Synthians services
2. **Error Handling** - Provide consistent error responses
3. **Data Transformation** - Format data for the frontend
4. **Authentication** - Manage user sessions (future implementation)

Key backend technologies include:

- **Express.js** - Web server framework
- **TypeScript** - Type-safe JavaScript
- **cors** - Cross-origin resource sharing

### Data Storage

The dashboard itself does not maintain persistent data storage but relies on the core services for data. In a development environment, it can use in-memory storage to simulate the services.

## Data Flow

1. **User Interaction** - User interacts with the React frontend
2. **API Request** - Frontend sends a request to the Express backend
3. **Service Proxying** - Backend forwards the request to the appropriate service
4. **Service Response** - Service processes the request and sends a response
5. **UI Update** - Frontend updates the UI based on the response

For real-time updates, the dashboard uses a combination of:

1. **Polling** - Regular API requests on a configurable interval
2. **WebSockets** - For log streaming and immediate notifications (future implementation)

## Key Design Decisions

### 1. Separation of Concerns

The dashboard is designed with a clear separation between different aspects of the application:

- **UI Components** - Presentation logic
- **API Hooks** - Data fetching logic
- **State Management** - Application state
- **Routing** - Navigation logic

This makes the codebase easier to maintain and test.

### 2. Type Safety

TypeScript is used throughout the application to ensure type safety and provide better developer experience. Shared schemas are defined in a central location to ensure consistency between frontend and backend.

### 3. Responsive Design

The dashboard is designed to be responsive and work well on different screen sizes. It uses a combination of:

- Responsive grid layouts
- Collapsible sidebar
- Adaptive components

### 4. Error Handling

The application has a comprehensive error handling strategy:

- API errors are caught and displayed to the user
- Network failures are gracefully handled
- Type errors are caught at compile time

### 5. Performance Optimization

Several strategies are used to optimize performance:

- Query caching with TanStack Query
- Memoization of expensive calculations
- Lazy loading of components
- Efficient rendering with React

## Future Architectural Considerations

1. **Authentication and Authorization** - Adding user accounts and role-based access control
2. **WebSocket Integration** - For real-time updates across all services
3. **Service Worker** - For offline support and caching
4. **Analytics** - For tracking usage patterns and performance metrics

## Development Guidelines

When extending the architecture, consider the following guidelines:

1. **Maintain Type Safety** - Add proper types for all new code
2. **Follow Component Structure** - Keep components small and focused
3. **Consistent State Management** - Use the existing state management patterns
4. **Document Changes** - Update this document and others when making architectural changesComponents
1. Frontend Client (/client)
Framework: React (using Vite for development and bundling).

Language: TypeScript.

UI Library: Shadcn UI built upon Radix UI and Tailwind CSS.

Routing: Wouter handles client-side navigation (/, /memory-core, /assemblies/:id, etc.).

State Management:

TanStack Query (@tanstack/react-query): Manages server state, caching, background refresh, and request status (loading, error) for data fetched from the dashboard's backend proxy (/api/...). Hooks are defined in client/src/lib/api.ts.

Zustand: Used for simple global client state, primarily the data polling interval (client/src/lib/store.ts).

Core Structure:

main.tsx: Entry point, sets up QueryClientProvider.

App.tsx: Defines routes using <Switch> and renders the main DashboardShell.

components/layout/: Contains DashboardShell, Sidebar, TopBar.

components/ui/: Contains Shadcn UI components.

components/dashboard/: Contains reusable components specific to this dashboard's views (e.g., OverviewCard, MetricsChart, ActivationExplanationView).

pages/: Contains top-level components for each route.

lib/: Utilities, API hooks (api.ts), Zustand store (store.ts), query client setup.

2. Backend Proxy Server (/server)
Framework: Express.js (running via Node.js).

Language: TypeScript (using tsx or compiled JS for execution).

Primary Role:

Serve Frontend: In development (via Vite middleware) and production (serving static build from /dist/public), it serves the index.html and associated assets.

API Proxying: All requests starting with /api/ are intercepted. The server determines the target backend service (MC, NM, CCE) based on the path (e.g., /api/memory-core/* proxies to MEMORY_CORE_URL). It uses axios to forward the request (method, query params, body) to the appropriate internal service URL (configured via environment variables). It then forwards the response (or error) back to the frontend client. This avoids CORS issues and hides the internal service URLs from the browser.

(Development/Mocking): Can include mock handlers for endpoints if backend services are unavailable (e.g., /api/alerts uses server/storage.ts).

Key Files:

server.mjs / server/index.ts: Entry point, sets up Express app, middleware, and Vite integration (dev only).

server/routes.ts: (CRITICAL) Defines the proxy routes. Needs significant updates for Phase 5.9.

server/vite.ts: Helper for Vite middleware integration.

server/storage.ts: Simple in-memory storage for mock data (e.g., alerts).

3. Synthians Core Services (External)
These are the independent backend services (Memory Core, Neural Memory, CCE) running, potentially in Docker containers.

The dashboard proxy needs the correct URLs (e.g., http://memory-core:5010) configured via environment variables (MEMORY_CORE_URL, etc.) to reach them.

Design Decisions
BFF Proxy: Simplifies frontend development by providing a single API endpoint (/api/...) and handling CORS. It can also potentially aggregate or cache data in the future.

TanStack Query: Provides robust caching, background refresh, and request state management, simplifying data fetching logic in components.

Shadcn UI & Tailwind: Offers a flexible and consistent design system based on unstyled primitives.

TypeScript: Enforces type safety across the frontend, backend proxy, and shared schemas.

Polling: Simple mechanism for periodic data refresh, managed by Zustand and TanStack Query invalidation. Real-time updates via WebSockets are a future enhancement.
```

# docs\CHANGELOG.md

```md
# Changelog

## [1.0.1] - 2025-04-05

### Fixed

- Added missing React imports to various components to fix "React is not defined" errors:
  - `App.tsx`
  - `DashboardShell.tsx`
  - `Sidebar.tsx`
  - `TopBar.tsx`
  - `toaster.tsx`
  - `skeleton.tsx`

- Fixed DOM nesting issues:
  - Changed nested `<a>` tags to `<div>` elements in `Sidebar.tsx` NavLink component
  - Changed nested `<a>` tags to `<div>` elements in `TopBar.tsx` Link component
  - Fixed type error in `Sidebar.tsx` by using `Boolean()` for conditional path checking

- Improved DashboardShell layout:
  - Enhanced mobile responsiveness
  - Fixed sidebar visibility in mobile and desktop views
  - Streamlined main content container structure

### Changed

- Updated `vite.config.ts` to use absolute paths for module aliases
- Removed conflicting JSX runtime options in configuration files
- Added proper cursor pointer styling to clickable elements

### Technical Details

- Path alias configuration in `vite.config.ts` was updated to avoid conflicts
- React 18 automatic JSX runtime is now properly utilized across components
- Invalid DOM nesting (nested `<a>` elements) resolved for better accessibility and standard compliance

```

# docs\DEVELOPMENT_GUIDE.md

```md
# Dashboard UI Components

This document provides an overview of the key React components used in the dashboard.

## Layout Components (`client/src/components/layout/`)

*   **`DashboardShell.tsx`:** The main application wrapper. Includes the `Sidebar` and `TopBar` and renders the main page content as children. Handles mobile sidebar toggling.
*   **`Sidebar.tsx`:** The left-hand navigation menu. Contains `NavLink` components for routing. Uses `wouter`'s `useLocation` to highlight the active page.
*   **`TopBar.tsx`:** The header bar. Includes a mobile sidebar toggle, search bar (placeholder), manual refresh button (`RefreshButton`), polling rate selector, and basic service status links.
*   **`ServiceStatus.tsx`:** A small component displaying the health status (Healthy, Unhealthy, etc.) of a backend service with a colored dot and text.

## UI Primitives (`client/src/components/ui/`)

*   These are standard components generated from **Shadcn UI**. They provide the building blocks for the interface (Buttons, Cards, Tables, Forms, Toasts, etc.). Refer to the Shadcn UI documentation for usage details.
*   Key components used extensively: `Card`, `Button`, `Table`, `Badge`, `Skeleton`, `Tabs`, `Select`, `Input`, `ScrollArea`, `Progress`, `Alert`, `Toast`, `Collapsible`.

## Dashboard Specific Components (`client/src/components/dashboard/`)

These components are tailored for displaying specific types of data within the dashboard views.

*   **`OverviewCard.tsx`:** Displays a summary card for a specific service (MC, NM, CCE), showing health status and key metrics.
*   **`MetricsChart.tsx`:** A reusable line chart component (using `Recharts`) for displaying time-series data (e.g., NM Loss/Grad Norm). Includes time range selection.
*   **`CCEChart.tsx`:** A specialized bar chart (using `Recharts`) for visualizing CCE variant distribution over time.
*   **`AssemblyTable.tsx`:** Displays a list of assemblies in a table format, including name, member count, update time, and **sync status**. Links to the detail view.
*   **`SystemArchitecture.tsx`:** Renders a static SVG-based diagram showing the high-level interaction between MC, NM, and CCE.
*   **`DiagnosticAlerts.tsx`:** Displays a list of recent alerts (currently mocked via `server/storage.ts`).

### Phase 5.9 Explainability Components:

*   **`ActivationExplanationView.tsx`:** Displays the detailed explanation for why a memory did or did not activate within an assembly. Renders data from the `useExplainActivation` hook.
*   **`MergeExplanationView.tsx`:** Displays the details of how an assembly was formed via a merge, including source assemblies, timestamp, and cleanup status. Renders data from the `useExplainMerge` hook.
*   **`LineageView.tsx`:** Displays the merge ancestry of an assembly in a list or tree-like format. Renders data from the `useAssemblyLineage` hook.
*   **`MergeLogView.tsx`:** Displays recent merge events fetched from the `/diagnostics/merge_log` endpoint via the `useMergeLog` hook. Correlates merge and cleanup events.

**Note:** These Phase 5.9 components currently exist but require the corresponding API hooks in `lib/api.ts` and proxy routes in `server/routes.ts` to be fully implemented to fetch and display real data.
```

# docs\FLOW_DIAGRAM.md

```md

--

7. Flow Diagram Example (Mermaid in docs/dashboard/FLOW_DIAGRAM.md)
# Dashboard Data Flow Diagrams

## Example: Explaining Assembly Merge

\`\`\`mermaid
sequenceDiagram
    participant User
    participant FE_Component as AssemblyDetail.tsx
    participant FE_Hook as useExplainMerge (api.ts)
    participant FE_Proxy as Dashboard Backend (routes.ts)
    participant BE_Service as Memory Core API

    User->>+FE_Component: Clicks "Explain Merge" Button
    FE_Component->>+FE_Hook: Calls explainMergeQuery.refetch()
    FE_Hook->>+FE_Proxy: GET /api/memory-core/assemblies/{id}/explain_merge
    Note over FE_Proxy: TODO: Implement this Proxy Route
    FE_Proxy->>+BE_Service: GET {MEMORY_CORE_URL}/assemblies/{id}/explain_merge
    BE_Service-->>-FE_Proxy: JSON Response (Merge Data or Error)
    FE_Proxy-->>-FE_Hook: Forward JSON Response
    FE_Hook-->>-FE_Component: TanStack Query updates data/error state
    FE_Component->>User: Renders MergeExplanationView with data or error
Use code with caution.
Markdown
Example: Loading Merge Log
sequenceDiagram
    participant User
    participant FE_Component as MergeLogPage.tsx (or similar)
    participant FE_Hook as useMergeLog (api.ts)
    participant FE_Proxy as Dashboard Backend (routes.ts)
    participant BE_Service as Memory Core API

    User->>+FE_Component: Navigates to Log Page
    FE_Component->>+FE_Hook: Renders component using useMergeLog(limit)
    Note over FE_Hook: TanStack Query automatically fetches on mount
    FE_Hook->>+FE_Proxy: GET /api/memory-core/diagnostics/merge_log?limit=50
    Note over FE_Proxy: TODO: Implement this Proxy Route
    FE_Proxy->>+BE_Service: GET {MEMORY_CORE_URL}/diagnostics/merge_log?limit=50
    BE_Service-->>-FE_Proxy: JSON Response (Log Entries or Error)
    FE_Proxy-->>-FE_Hook: Forward JSON Response
    FE_Hook-->>-FE_Component: TanStack Query provides data/state
    FE_Component->>User: Renders MergeLogView with data
Use code with caution.
Mermaid
(Add similar diagrams for other key interactions like fetching stats, config, lineage, activation explanation)

---

This documentation suite provides a solid foundation for understanding the dashboard project and tackling the Phase 5.9 integration work. Remember to update the **TODO** sections in the actual code (`server/routes.ts`, `client/src/lib/api.ts`, `shared/schema.ts`) as you implement the necessary connections.
```

# docs\PROJECT_STRUCTURE.md

```md

---

## **3. `docs/dashboard/PROJECT_STRUCTURE.md` (File Tree & Components)**

\`\`\`markdown
# Dashboard Project Structure

This document outlines the file and directory structure of the Synthians Cognitive Dashboard project.

## File Tree Diagram

\`\`\`plaintext
Synthians_dashboard/
├── client/                   # React Frontend Application
│   ├── public/
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── dashboard/      # Dashboard-specific complex components
│   │   │   │   ├── ActivationExplanationView.tsx
│   │   │   │   ├── AssemblyTable.tsx
│   │   │   │   ├── CCEChart.tsx
│   │   │   │   ├── DiagnosticAlerts.tsx
│   │   │   │   ├── LineageView.tsx
│   │   │   │   ├── MergeExplanationView.tsx
│   │   │   │   ├── MergeLogView.tsx
│   │   │   │   ├── MetricsChart.tsx
│   │   │   │   ├── OverviewCard.tsx
│   │   │   │   └── SystemArchitecture.tsx
│   │   │   ├── layout/         # Core layout components
│   │   │   │   ├── DashboardShell.tsx
│   │   │   │   ├── ServiceStatus.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── TopBar.tsx
│   │   │   └── ui/             # Shadcn UI components (Button, Card, etc.)
│   │   │       ├── accordion.tsx
│   │   │       ├── ... (all shadcn components) ...
│   │   │       └── tooltip.tsx
│   │   ├── hooks/            # Custom React hooks
│   │   │   ├── use-mobile.tsx
│   │   │   └── use-toast.ts
│   │   ├── lib/              # Utilities and core logic
│   │   │   ├── api.ts          # TanStack Query hooks for API calls (NEEDS 5.9 UPDATES)
│   │   │   ├── queryClient.ts  # TanStack Query client configuration
│   │   │   ├── store.ts        # Zustand stores (polling, theme)
│   │   │   └── utils.ts        # Utility functions (e.g., cn)
│   │   ├── pages/            # Route components
│   │   │   ├── admin.tsx
│   │   │   ├── assemblies/
│   │   │   │   ├── index.tsx     # Assembly list view
│   │   │   │   └── [id].tsx      # Assembly detail/inspector view (OLD - replaced by assembly-inspector.tsx?)
│   │   │   ├── assembly-inspector.tsx # (NEW - Preferred name for detail view)
│   │   │   ├── cce.tsx
│   │   │   ├── chat.tsx        # Placeholder
│   │   │   ├── config.tsx
│   │   │   ├── logs.tsx        # Placeholder (could show merge log)
│   │   │   ├── llm-guidance.tsx
│   │   │   ├── memory-core.tsx
│   │   │   ├── neural-memory.tsx
│   │   │   ├── not-found.tsx
│   │   │   └── overview.tsx
│   │   ├── App.tsx             # Main application component with routing
│   │   ├── index.css           # Tailwind CSS base/styles
│   │   └── main.tsx            # Application entry point
│   ├── index.html          # Main HTML file
│   └── vite-env.d.ts       # Vite TypeScript definitions
├── server/                   # Express Backend Proxy Server
│   ├── index.ts            # Server entry point
│   ├── routes.ts           # API proxy route definitions (NEEDS 5.9 UPDATES)
│   ├── storage.ts          # In-memory storage for mocking (e.g., alerts)
│   ├── vite.ts             # Vite middleware integration helper
│   └── package.json        # Server dependencies
├── shared/                   # Shared TypeScript types/schemas
│   └── schema.ts           # Defines API data structures (NEEDS 5.9 UPDATES)
├── docs/                     # Project documentation (like this file)
│   └── dashboard/
├── attached_assets/          # (Potentially unused assets?)
├── .gitignore
├── .replit                   # Replit configuration
├── CONTRIBUTING.md
├── dev.ps1                   # Development start script (Windows PowerShell)
├── drizzle.config.ts         # Drizzle ORM config (may not be fully used yet)
├── generated-icon.png        # (Likely generated by Replit)
├── package.json              # Main project dependencies & scripts
├── postcss.config.js
├── README.md                 # Top-level project README
├── server.mjs                # Alternative/simplified server entry point?
├── start-dev.js              # Development start script (Node)
├── tailwind.config.ts
├── theme.json                # Shadcn theme config
└── tsconfig.json             # TypeScript configuration

Key Component Overview
DashboardShell: Provides the main layout including Sidebar and TopBar.

Sidebar: Contains navigation links defined via NavLink components. Uses wouter's useLocation to highlight the active link.

TopBar: Includes search (placeholder), refresh button, polling controls, and basic status indicators.

Page Components (pages/): Each corresponds to a route in App.tsx. They fetch data using hooks from lib/api.ts and render specific dashboard/ or ui/ components.

api.ts: Central place for defining useQuery hooks that interact with the dashboard's backend proxy API (/api/...). Needs functions/hooks for Phase 5.9 data.

routes.ts: Defines the Express routes on the dashboard's server. Needs proxy routes for Phase 5.9 backend endpoints.

schema.ts: Needs TypeScript interfaces matching the Pydantic models defined in docs/api/phase_5_9_models.md.

Explainability Components (dashboard/): ActivationExplanationView, MergeExplanationView, LineageView, MergeLogView are present but rely on data fetched via hooks in api.ts that need to target the (currently missing) proxy routes for Phase 5.9 endpoints.

---

## **4. `docs/dashboard/DATA_FLOW_API.md` (Data Flow & TODOs)**

\`\`\`markdown
# Dashboard Data Flow & API Integration

This document explains how the Synthians Cognitive Dashboard fetches and displays data, highlighting the necessary steps to integrate Phase 5.9 backend features.

## Data Fetching Architecture

The dashboard uses a tiered approach for fetching data:

1.  **React Component:** A component (e.g., `pages/memory-core.tsx`) needs data.
2.  **TanStack Query Hook:** The component calls a custom hook from `client/src/lib/api.ts` (e.g., `useMemoryCoreStats()`).
3.  **`useQuery`:** This hook uses TanStack Query's `useQuery`. The `queryKey` typically represents the API path relative to the proxy (e.g., `['/api/memory-core/stats']`).
4.  **API Request:** `useQuery`'s `queryFn` (configured in `lib/queryClient.ts` or `lib/api.ts`) uses `axios` to make an HTTP request to the dashboard's **backend proxy server** (e.g., `GET http://localhost:5000/api/memory-core/stats`).
5.  **Proxy Forwarding:** The dashboard's Express server (`server/routes.ts`) intercepts the `/api/...` request. It identifies the target service (e.g., Memory Core) and forwards the request using `axios` to the actual service URL (e.g., `http://memory-core:5010/stats`).
6.  **Service Response:** The target Synthians service (e.g., Memory Core) processes the request and sends back JSON data.
7.  **Proxy Response:** The dashboard's Express server receives the response and forwards it back to the frontend client.
8.  **TanStack Query Cache:** TanStack Query receives the data, updates its cache, and makes the data available to the React component via the hook.
9.  **Component Render:** The React component re-renders with the fetched data, loading, or error state.

## Polling & Refreshing

*   **Polling:** The `usePollingStore` (Zustand) manages a global interval timer. On each tick, it calls `refreshAllData()` from `lib/api.ts`.
*   **`refreshAllData()`:** This function uses TanStack Query's `queryClient.invalidateQueries()` to mark relevant queries as stale, triggering background refetches for updated data.
*   **Manual Refresh:** The `<RefreshButton />` in the `TopBar` also calls `refreshAllData()`.

## API Proxy (`server/routes.ts`)

This is the **critical integration point** that bridges the frontend and the actual backend services.

**Current Status (Needs Update for 5.9):**

*   Proxies exist for basic health, status, and stats endpoints for MC, NM, CCE.
*   Proxies exist for listing/getting assemblies (`/api/memory-core/assemblies`).
*   Admin action proxies exist (verify index, set variant, etc.).
*   Mock `/api/alerts` endpoint exists.

**❗ Phase 5.9 TODOs - Add Proxy Routes in `server/routes.ts`:**

\`\`\`typescript
// Example Structure (add these within registerRoutes in server/routes.ts)

// --- Memory Core Explainability Proxies ---
apiRouter.get("/memory-core/assemblies/:id/explain_activation", /* ... proxy to MC ... */ );
apiRouter.get("/memory-core/assemblies/:id/explain_merge",    /* ... proxy to MC ... */ );
apiRouter.get("/memory-core/assemblies/:id/lineage",         /* ... proxy to MC ... */ );

// --- Memory Core Diagnostics Proxies ---
apiRouter.get("/memory-core/diagnostics/merge_log",       /* ... proxy to MC (forward limit param) ... */ );
apiRouter.get("/memory-core/config/runtime/:service",   /* ... proxy to MC (use service param) ... */ );

// --- (Optional) Proxies for NM/CCE Runtime Config (if MC doesn't handle them) ---
// apiRouter.get("/neural-memory/config/runtime", /* ... proxy to NM or MC ... */ );
// apiRouter.get("/cce/config/runtime",           /* ... proxy to CCE or MC ... */ );

Ensure these proxy handlers correctly forward path parameters (:id, :service), query parameters (?memory_id=..., ?limit=...), and handle errors (forwarding status codes like 404, 403, 500).

API Client Hooks (client/src/lib/api.ts)
This file defines useQuery hooks for easy data fetching in components.

Current Status (Needs Update for 5.9):

Hooks exist for basic health, status, stats, assemblies.

❗ Phase 5.9 TODOs - Add useQuery Hooks in client/src/lib/api.ts:

// Example Structure (add these hooks to client/src/lib/api.ts)
import { /* Import necessary response types from @shared/schema */ } from '@shared/schema';

// --- Explainability Hooks ---
export const useExplainActivation = (assemblyId: string | null, memoryId?: string | null) => {
  return useQuery<ExplainActivationResponse>({ // Use correct response type
    queryKey: ['memory-core', 'assemblies', assemblyId, 'explain_activation', { memory_id: memoryId }],
    queryFn: defaultQueryFn, // Or custom fetcher
    enabled: !!assemblyId && !!memoryId, // Only enable when IDs are present
    retry: 1,
    staleTime: Infinity, // Data likely won't change unless manually triggered
  });
};

export const useExplainMerge = (assemblyId: string | null) => {
  return useQuery<ExplainMergeResponse>({ // Use correct response type
    queryKey: ['memory-core', 'assemblies', assemblyId, 'explain_merge'],
    queryFn: defaultQueryFn,
    enabled: !!assemblyId,
    retry: 1,
    staleTime: Infinity,
  });
};

export const useAssemblyLineage = (assemblyId: string | null) => {
  return useQuery<LineageResponse>({ // Use correct response type
    queryKey: ['memory-core', 'assemblies', assemblyId, 'lineage'],
    queryFn: defaultQueryFn,
    enabled: !!assemblyId,
    retry: 1,
    staleTime: Infinity,
  });
};

// --- Diagnostics Hooks ---
export const useMergeLog = (limit: number = 50) => {
  return useQuery<MergeLogResponse>({ // Use correct response type
    queryKey: ['memory-core', 'diagnostics', 'merge_log', { limit }],
    queryFn: defaultQueryFn,
    refetchInterval: 30000, // Optionally refetch merge log periodically
  });
};

export const useRuntimeConfig = (serviceName: string | null) => {
  return useQuery<RuntimeConfigResponse>({ // Use correct response type
    queryKey: ['memory-core', 'config', 'runtime', serviceName], // Assumes MC proxies all
    queryFn: defaultQueryFn,
    enabled: !!serviceName,
    staleTime: 5 * 60 * 1000, // Config changes less often, longer stale time
  });
};

// --- Update refreshAllData ---
// Add invalidations for new query keys like merge_log
// queryClient.invalidateQueries({ queryKey: ['memory-core', 'diagnostics', 'merge_log'] });
Use code with caution.
TypeScript
Ensure correct queryKey structures are used.

Use appropriate enabled flags (e.g., only fetch details when an ID is present, disable explain hooks by default until manually triggered).

Import and use the correct TypeScript response types from @shared/schema.ts.

Update refreshAllData to invalidate new queries if needed.

Shared Schema (shared/schema.ts)
❗ Phase 5.9 TODOs - Define TypeScript Interfaces:

Add interfaces matching the Pydantic models for all new API responses:

ExplainActivationResponse (containing ExplainActivationData)

ExplainMergeResponse (containing ExplainMergeData)

LineageResponse (containing LineageEntry[])

MergeLogResponse (containing MergeLogEntry[])

RuntimeConfigResponse (containing config: Record<string, any>)

Ensure existing types (Assembly, MemoryStats) are up-to-date if the backend changed them.
```

# docs\QUICK_START.md

```md
# Synthians Cognitive Dashboard - Quick Start Guide

This guide will help you quickly set up the Synthians Cognitive Dashboard for local development.

## Prerequisites

- Node.js 20.x or higher
- npm 9.x or higher
- Git

## Setup Steps

### 1. Clone the Repository

\`\`\`bash
git clone https://github.com/synthians/cognitive-dashboard.git
cd cognitive-dashboard
\`\`\`

### 2. Install Dependencies

\`\`\`bash
npm install
\`\`\`

### 3. Configure Environment

Create a `.env` file in the root directory with the following variables:

\`\`\`
# Core Service URLs
MEMORY_CORE_URL=http://localhost:8080
NEURAL_MEMORY_URL=http://localhost:8081
CCE_URL=http://localhost:8082

# Development Settings
NODE_ENV=development
PORT=5000
\`\`\`

### 4. Start the Development Server

\`\`\`bash
npm run dev
\`\`\`

This will start both the Express backend server and the frontend development server. The application will be available at `http://localhost:5000`.

## Project Structure

\`\`\`
/client             # Frontend React application
  /src
    /components     # UI components
    /hooks          # Custom React hooks
    /lib            # Utilities and API clients
    /pages          # Page components
/server             # Express backend
  /routes.ts        # API routes
  /storage.ts       # Storage interfaces
/shared             # Shared TypeScript schemas
\`\`\`

## Key Development Workflows

### Adding a New Dashboard Page

1. Create a new page component in `client/src/pages/`
2. Add the route in `client/src/App.tsx`
3. Add a sidebar navigation link in `client/src/components/layout/Sidebar.tsx`

Example page component:

\`\`\`tsx
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function NewFeature() {
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">New Feature</h2>
          <p className="text-sm text-gray-400">
            Description of the new feature
          </p>
        </div>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Feature Details</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Feature content goes here */}
        </CardContent>
      </Card>
    </>
  );
}
\`\`\`

### Adding a New API Endpoint

1. Add the endpoint in `server/routes.ts`
2. Create a client-side API hook in `client/src/lib/api.ts`

Example API endpoint:

\`\`\`typescript
// In server/routes.ts
app.get("/api/new-feature/data", (req, res) => {
  // Implementation
  res.json({ data: { /* your data */ } });
});

// In client/src/lib/api.ts
export const useNewFeatureData = () => {
  return useQuery({
    queryKey: ["/api/new-feature/data"],
    staleTime: 30000
  });
};
\`\`\`

### Working with Mock Data During Development

While developing, you may need to work with mock data before connecting to real services:

1. Create mock handlers in the server routes
2. Use consistent data structures based on the shared schema

Example mock implementation:

\`\`\`typescript
// In server/routes.ts
app.get("/api/memory-core/assemblies", (req, res) => {
  // Return mock data following the Assembly schema
  res.json({
    data: [
      {
        id: "assembly-1",
        name: "Test Assembly",
        description: "A test assembly for development",
        member_count: 42,
        keywords: ["test", "development"],
        tags: ["important"],
        topics: ["testing"],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        memory_ids: ["mem-1", "mem-2"]
      }
    ]
  });
});
\`\`\`

## Troubleshooting

### API Connection Issues

If you're having trouble connecting to the core services:

1. Check that the environment variables are correctly set
2. Verify that the services are running on the expected ports
3. Check for CORS issues in the browser dev tools

### Build Errors

If you encounter build errors:

1. Check for TypeScript errors
2. Ensure all dependencies are installed
3. Clear the node_modules folder and reinstall

\`\`\`bash
rm -rf node_modules
npm install
\`\`\`

## Next Steps

After setting up your development environment, you might want to:

1. Explore the existing codebase to understand the architecture
2. Check out the open issues for potential contributions
3. Run the test suite to ensure everything is working correctly

For more detailed information, refer to the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) files.
```

# docs\README.md

```md
# Synthians Cognitive Dashboard - Documentation

Welcome to the documentation for the Synthians Cognitive Dashboard project.

## Overview

This project implements a web-based user interface for monitoring, inspecting, and interacting with the Synthians Cognitive Architecture services (Memory Core, Neural Memory, CCE). It aims to provide real-time visibility into system health, performance metrics, internal states (like memory assemblies and merge history), configuration, and includes placeholders for future interactive features like live logging and chat.

**Phase Context:** This documentation describes the dashboard structure as planned for integration with the **Phase 5.9 backend features** (Explainability & Diagnostics). Many UI components are present, but the connections to the specific Phase 5.9 backend APIs are **TODO** items.

## Key Features (Planned & Partially Implemented)

*   **Service Status Monitoring:** Health, uptime, version for MC, NM, CCE.
*   **Core Metrics Display:** Memory/assembly counts, vector index stats, NM performance (loss/grad), CCE variant selection.
*   **Assembly Inspector:** Browse assemblies, view details, members, metadata, and **planned explainability views (lineage, activation, merge)**.
*   **Configuration Viewer:** Display **sanitized** runtime configurations from services.
*   **Diagnostics Views:** Display **merge log history**.
*   **(Placeholders):** Real-time Log Streaming, Interactive Chat, Admin Actions.

## Technology Stack

*   **Frontend:** React (Vite), TypeScript, Tailwind CSS, Shadcn UI
*   **Routing:** Wouter
*   **State Management:** TanStack Query (Server State), Zustand (Client State - e.g., polling)
*   **Charting:** Recharts
*   **Backend (Dashboard Proxy):** Express.js (Node.js), TypeScript, Axios (for proxying)
*   **(Optional for Dev):** In-memory storage for mocking alerts.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** Dashboard's internal architecture (Client, Proxy Backend).
*   **[Project Structure](./PROJECT_STRUCTURE.md):** File tree and component overview.
*   **[Data Flow & API](./DATA_FLOW_API.md):** How data is fetched via the proxy backend, with **TODOs** for Phase 5.9 integration.
*   **[Development Guide](./DEVELOPMENT_GUIDE.md):** Setup, running, adding features, best practices.
*   **[UI Components](./UI_COMPONENTS.md):** Overview of key layout and dashboard-specific components.

## Getting Started

Refer to the **[Development Guide](./DEVELOPMENT_GUIDE.md)** for setup and running instructions.
```

# drizzle.config.ts

```ts
import { defineConfig } from "drizzle-kit";

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL, ensure the database is provisioned");
}

export default defineConfig({
  out: "./migrations",
  schema: "./shared/schema.ts",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL,
  },
});

```

# generated-icon.png

This is a binary file of the type: Image

# package.json

```json
{
  "name": "synthians-cognitive-dashboard",
  "version": "1.0.0",
  "type": "module",
  "license": "MIT",
  "scripts": {
    "dev": "node server.mjs",
    "start": "NODE_ENV=production node dist/index.js",
    "build": "vite build && esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist",
    "check": "tsc",
    "db:push": "drizzle-kit push"
  },
  "dependencies": {
    "@hookform/resolvers": "^3.9.1",
    "@jridgewell/trace-mapping": "^0.3.25",
    "@neondatabase/serverless": "^0.10.4",
    "@radix-ui/react-accordion": "^1.2.1",
    "@radix-ui/react-alert-dialog": "^1.1.2",
    "@radix-ui/react-aspect-ratio": "^1.1.0",
    "@radix-ui/react-avatar": "^1.1.1",
    "@radix-ui/react-checkbox": "^1.1.2",
    "@radix-ui/react-collapsible": "^1.1.1",
    "@radix-ui/react-context-menu": "^2.2.2",
    "@radix-ui/react-dialog": "^1.1.2",
    "@radix-ui/react-dropdown-menu": "^2.1.2",
    "@radix-ui/react-hover-card": "^1.1.2",
    "@radix-ui/react-label": "^2.1.0",
    "@radix-ui/react-menubar": "^1.1.2",
    "@radix-ui/react-navigation-menu": "^1.2.1",
    "@radix-ui/react-popover": "^1.1.2",
    "@radix-ui/react-progress": "^1.1.0",
    "@radix-ui/react-radio-group": "^1.2.1",
    "@radix-ui/react-scroll-area": "^1.2.0",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-separator": "^1.1.0",
    "@radix-ui/react-slider": "^1.2.1",
    "@radix-ui/react-slot": "^1.1.0",
    "@radix-ui/react-switch": "^1.1.1",
    "@radix-ui/react-tabs": "^1.1.1",
    "@radix-ui/react-toast": "^1.2.2",
    "@radix-ui/react-toggle": "^1.1.0",
    "@radix-ui/react-toggle-group": "^1.1.0",
    "@radix-ui/react-tooltip": "^1.1.3",
    "@replit/vite-plugin-shadcn-theme-json": "^0.0.4",
    "@tanstack/react-query": "^5.60.5",
    "axios": "^1.8.4",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "cmdk": "^1.0.0",
    "connect-pg-simple": "^10.0.0",
    "date-fns": "^3.6.0",
    "drizzle-orm": "^0.39.1",
    "drizzle-zod": "^0.7.0",
    "embla-carousel-react": "^8.3.0",
    "express": "^4.21.2",
    "express-session": "^1.18.1",
    "framer-motion": "^11.13.1",
    "input-otp": "^1.2.4",
    "lucide-react": "^0.453.0",
    "memorystore": "^1.6.7",
    "passport": "^0.7.0",
    "passport-local": "^1.0.0",
    "react": "^18.3.1",
    "react-day-picker": "^8.10.1",
    "react-dom": "^18.3.1",
    "react-hook-form": "^7.53.1",
    "react-icons": "^5.4.0",
    "react-resizable-panels": "^2.1.4",
    "recharts": "^2.13.0",
    "tailwind-merge": "^2.5.4",
    "tailwindcss-animate": "^1.0.7",
    "vaul": "^1.1.0",
    "wouter": "^3.3.5",
    "ws": "^8.18.0",
    "zod": "^3.23.8",
    "zod-validation-error": "^3.4.0",
    "zustand": "^5.0.3"
  },
  "devDependencies": {
    "@replit/vite-plugin-cartographer": "^0.0.11",
    "@replit/vite-plugin-runtime-error-modal": "^0.0.3",
    "@tailwindcss/typography": "^0.5.15",
    "@types/connect-pg-simple": "^7.0.3",
    "@types/express": "4.17.21",
    "@types/express-session": "^1.18.0",
    "@types/node": "20.16.11",
    "@types/passport": "^1.0.16",
    "@types/passport-local": "^1.0.38",
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.1",
    "@types/ws": "^8.5.13",
    "@vitejs/plugin-react": "^4.3.2",
    "autoprefixer": "^10.4.20",
    "drizzle-kit": "^0.30.4",
    "esbuild": "^0.25.0",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "tsx": "^4.19.1",
    "typescript": "5.6.3",
    "vite": "^5.4.14"
  },
  "optionalDependencies": {
    "bufferutil": "^4.0.8"
  }
}

```

# postcss.config.js

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}

```

# README.md

```md
# Synthians Cognitive Architecture Development Dashboard

The Synthians Cognitive Architecture Development Dashboard is a comprehensive web-based monitoring and management interface for the Synthians AI system. This dashboard provides real-time visibility, diagnostic capabilities, and interactive interfaces for core components of the cognitive architecture.

![Synthians Development Dashboard](./docs/images/dashboard-preview.png)

## 🧠 Features

- **System Overview**: Real-time monitoring of core services health and performance metrics
- **Component Dashboards**: Detailed views for Memory Core, Neural Memory, and Controlled Context Exchange (CCE)
- **Assembly Inspector**: Browse and analyze memory assemblies and their relationships
- **Real-time Logs**: Stream and filter logs from all system components
- **Admin Controls**: Maintenance and configuration actions for system components
- **Interactive Chat**: Directly engage with the Synthians AI through a chat interface
- **Configuration Management**: View and modify system configuration parameters
- **LLM Guidance Monitoring**: Track interactions with external LLM services

## 💻 Tech Stack

- **Frontend**: React, TypeScript, TailwindCSS, Shadcn UI
- **State Management**: TanStack Query, Zustand
- **Data Visualization**: Recharts
- **Backend**: Express.js, TypeScript
- **API Integration**: REST APIs to Synthians core services

## 🚀 Getting Started

### Prerequisites

- Node.js 20.x or higher
- npm 9.x or higher
- Access to the Synthians Cognitive Architecture services (Memory Core, Neural Memory, CCE)

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/synthians/cognitive-dashboard.git
cd cognitive-dashboard
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
\`\`\`

3. Configure environment variables:
\`\`\`bash
cp .env.example .env
\`\`\`

Edit the `.env` file to include the addresses of your Synthians Cognitive Architecture services:

\`\`\`
MEMORY_CORE_URL=http://localhost:8080
NEURAL_MEMORY_URL=http://localhost:8081
CCE_URL=http://localhost:8082
\`\`\`

4. Start the development server:
\`\`\`bash
npm run dev
\`\`\`

The dashboard will be available at `http://localhost:5000`

## 🏗️ Architecture

The Synthians Cognitive Dashboard follows a client-server architecture:

### Client

The client is a React application with the following key features:
- **Component-based Structure**: Modular components for different dashboard elements
- **Real-time Data**: Uses TanStack Query for efficient data fetching and caching
- **Responsive Design**: Mobile-friendly layout with adaptive components

### Server

The server is an Express.js application that:
- Serves the frontend application
- Proxies API requests to the core Synthians services
- Provides authentication and session management
- Handles data transformation and aggregation

### Core Services

The dashboard integrates with three primary services:

1. **Memory Core**: Manages episodic and semantic memory storage and retrieval
2. **Neural Memory**: Handles vector embedding generation and maintenance
3. **CCE (Controlled Context Exchange)**: Orchestrates information flow between components

## 📁 Project Structure

\`\`\`
/client             # Frontend application
  /src
    /components     # Reusable UI components
    /hooks          # Custom React hooks
    /lib            # Utilities and API clients
    /pages          # Page components
/server             # Backend Express server
  /routes           # API route handlers
  /storage          # Storage interfaces
/shared             # Shared TypeScript schemas
/docs               # Documentation
\`\`\`

## 🧩 Core Components

### Memory Core

The Memory Core dashboard provides visibility into:
- Memory storage statistics
- Vector index health
- Assembly metrics and status
- Memory retrieval performance

### Neural Memory

The Neural Memory dashboard displays:
- Training status and metrics
- Emotional loop diagnostics
- Vector embedding quality metrics
- Runtime configuration

### CCE (Controlled Context Exchange)

The CCE dashboard shows:
- Active variant information
- Response metrics
- LLM guidance statistics
- Performance indicators

## 🔧 Development

### Adding New Features

1. For frontend changes:
   - Add components to `client/src/components`
   - Add pages to `client/src/pages`
   - Update routing in `client/src/App.tsx`

2. For backend changes:
   - Add API routes to `server/routes.ts`
   - Update storage interfaces in `server/storage.ts`
   - Add schema definitions to `shared/schema.ts`

### Code Style

- Follow TypeScript best practices
- Use functional components with hooks for React code
- Add comprehensive comments for complex logic
- Include type definitions for all functions and variables

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

- [Shadcn UI](https://ui.shadcn.com/) for component primitives
- [TailwindCSS](https://tailwindcss.com/) for styling
- [TanStack Query](https://tanstack.com/query) for data fetching
- [Recharts](https://recharts.org/) for visualization components
```

# server.mjs

```mjs
// server.mjs - Modern ESM entry point for the Synthians Cognitive Dashboard
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import express from 'express';
import { createServer } from 'http';
import { createServer as createViteServer } from 'vite';

// Set environment for development
process.env.NODE_ENV = 'development';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Create Express app
const app = express();

// Body parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Create routes directly instead of importing from routes.ts
async function setupSimpleRoutes() {
  // Create a simple router and server for testing
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Simple server running' });
  });
  
  app.get('/api/test', (req, res) => {
    res.json({ message: 'API test endpoint working' });
  });

  // Simplified memory core routes
  app.get('/api/memory-core/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/api/memory-core/stats', (req, res) => {
    // Return mock stats data based on Phase 5.8 memory assembly stats requirements
    res.json({
      assemblies: {
        count: 24,
        average_size: 8,
        activation_count: 156,
        pending_updates: 0
      },
      memories: {
        count: 512,
        by_type: {
          declarative: 320,
          procedural: 125,
          episodic: 67
        }
      },
      system: {
        uptime: "3h 22m",
        version: "0.9.5-beta"
      }
    });
  });

  // Phase 5.9 explainability endpoints
  app.get('/api/memory-core/assemblies/:id/explain_activation', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      explanation: "This assembly was activated because Memory #M-12345 matched the input query with similarity score 0.87 (threshold: 0.75).",
      activation_details: {
        memory_id: "M-12345",
        similarity_score: 0.87,
        threshold: 0.75,
        activated_at: "2025-04-05T11:58:22Z"
      }
    });
  });

  app.get('/api/memory-core/assemblies/:id/explain_merge', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      explanation: "This assembly was formed by merging 3 source assemblies based on semantic similarity.",
      merge_details: {
        source_assemblies: ["ASM-001", "ASM-002", "ASM-005"],
        similarity_threshold: 0.82,
        merge_time: "2025-04-05T10:12:45Z",
        cleanup_status: "completed"
      }
    });
  });

  app.get('/api/memory-core/assemblies/:id/lineage', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      lineage: [
        {
          level: 0,
          assembly_id: req.params.id,
          created_at: "2025-04-05T10:12:45Z",
          merge_source: "direct_merge"
        },
        {
          level: 1,
          assembly_id: "ASM-001",
          created_at: "2025-04-05T09:35:12Z",
          merge_source: "direct_creation"
        },
        {
          level: 1,
          assembly_id: "ASM-002",
          created_at: "2025-04-05T08:22:31Z",
          merge_source: "direct_creation"
        },
        {
          level: 1,
          assembly_id: "ASM-005",
          created_at: "2025-04-05T07:45:19Z",
          merge_source: "previous_merge"
        }
      ]
    });
  });

  app.get('/api/diagnostics/merge_log', (req, res) => {
    res.json({
      entries: [
        {
          merge_event_id: "merge-123",
          timestamp: "2025-04-05T11:12:45Z",
          source_assembly_ids: ["ASM-007", "ASM-009"],
          result_assembly_id: "ASM-012",
          similarity_score: 0.85,
          threshold_used: 0.8,
          cleanup_status: "completed"
        },
        {
          merge_event_id: "merge-122",
          timestamp: "2025-04-05T10:55:32Z",
          source_assembly_ids: ["ASM-003", "ASM-004"],
          result_assembly_id: "ASM-011",
          similarity_score: 0.91,
          threshold_used: 0.8,
          cleanup_status: "completed"
        },
        {
          merge_event_id: "merge-121",
          timestamp: "2025-04-05T10:22:18Z",
          source_assembly_ids: ["ASM-001", "ASM-002", "ASM-005"],
          result_assembly_id: "ASM-010",
          similarity_score: 0.83,
          threshold_used: 0.8,
          cleanup_status: "failed",
          error: "Timeout during vector index update"
        }
      ]
    });
  });

  app.get('/api/neural-memory/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/api/cce/health', (req, res) => {
    res.json({ status: 'ok' });
  });
  
  const server = createServer(app);
  return server;
}

// Setup Vite for the frontend
async function setupVite(server) {
  try {
    // Create Vite server with proper path resolution for @ alias
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: { server },
      },
      // Use root directory to match our vite.config.ts
      root: resolve(__dirname, 'client'),
      // Configure path aliases - must match vite.config.ts
      resolve: {
        alias: {
          '@': resolve(__dirname, 'client/src'),
          '@shared': resolve(__dirname, 'shared'),
          '@assets': resolve(__dirname, 'attached_assets')
        }
      },
      // When using Windows paths, ensure proper path resolution
      appType: 'spa',
      optimizeDeps: {
        include: [
          'react',
          'react-dom',
          '@radix-ui/react-toast',
          'class-variance-authority',
          'clsx',
          'tailwind-merge'
        ]
      }
    });

    // Use Vite's connect instance as middleware
    app.use(vite.middlewares);

    // Handle all non-API routes with Vite
    app.use('*', async (req, res, next) => {
      // Skip API routes
      if (req.originalUrl.startsWith('/api')) {
        return next();
      }

      try {
        // Serve index.html through Vite's transform for all non-API routes
        const url = req.originalUrl;
        const indexPath = resolve(__dirname, 'client', 'index.html');

        // Transform the index.html with proper React imports
        let template = await vite.transformIndexHtml(url, `
          <!DOCTYPE html>
          <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1" />
              <link rel="icon" type="image/ico" href="/favicon.ico" />
              <title>Synthians Cognitive Dashboard</title>
            </head>
            <body>
              <div id="root"></div>
              <script type="module" src="/src/main.tsx"></script>
            </body>
          </html>
        `);
        
        res.status(200).set({ 'Content-Type': 'text/html' }).end(template);
      } catch (error) {
        console.error('Error serving frontend:', error);
        vite.ssrFixStacktrace(error);
        res.status(500).send('Internal Server Error');
      }
    });

    console.log('Vite middleware configured successfully');
  } catch (error) {
    console.error('Failed to initialize Vite middleware:', error);
  }
}

// Start server
async function startServer() {
  console.log('Starting Synthians Cognitive Dashboard server...');
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  try {
    // Create simplified routes for testing
    const server = await setupSimpleRoutes();
    
    // Setup Vite for frontend
    await setupVite(server);
    
    // Start server - use a different port (5500) to avoid conflicts
    const PORT = process.env.PORT || 5500;
    server.listen(PORT, () => {
      console.log(`Server running on port ${PORT} in ${process.env.NODE_ENV} mode`);
      console.log(`Dashboard available at http://localhost:${PORT}`);
      console.log(`Test API at http://localhost:${PORT}/api/health`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

```

# server\index.ts

```ts
import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on port 5000
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = 5000;
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true,
  }, () => {
    log(`serving on port ${port}`);
  });
})();

```

# server\package.json

```json
{
  "name": "server",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "description": "",
  "dependencies": {
    "axios": "^1.8.4",
    "cors": "^2.8.5",
    "dotenv": "^16.4.7",
    "express": "^5.1.0",
    "http-proxy-middleware": "^3.0.3"
  },
  "devDependencies": {
    "@types/cors": "^2.8.17",
    "@types/express": "^5.0.1",
    "@types/node": "^22.14.0",
    "nodemon": "^3.1.9",
    "ts-node": "^10.9.2",
    "typescript": "^5.8.3"
  }
}

```

# server\routes.ts

```ts
import express, { Router, Request, Response, RequestHandler } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import axios from "axios";

// Define API endpoints for the various services
const MEMORY_CORE_URL = process.env.MEMORY_CORE_URL || "http://memory-core:8080";
const NEURAL_MEMORY_URL = process.env.NEURAL_MEMORY_URL || "http://neural-memory:8080";
const CCE_URL = process.env.CCE_URL || "http://cce:8080";

export async function registerRoutes(app: express.Express): Promise<Server> {
  // Create a router instance for API routes
  const apiRouter = Router();
  
  // Memory Core routes
  apiRouter.get("/memory-core/health", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to Memory Core service" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/stats", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/stats`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Memory Core stats" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch assemblies" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch assembly" });
        }
      });
  }) as RequestHandler);

  // Phase 5.9 Explainability endpoints
  apiRouter.get("/memory-core/assemblies/:id/lineage", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/lineage`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch assembly lineage" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_merge", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/explain_merge`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found or no merge data available" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch merge explanation" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_activation", ((req: Request, res: Response) => {
    const memory_id = req.query.memory_id;
    if (!memory_id) {
      return res.status(400).json({ status: "Error", message: "memory_id parameter is required" });
    }
    
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/explain_activation`, { params: { memory_id } })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly or memory not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch activation explanation" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/diagnostics/merge_log", ((req: Request, res: Response) => {
    const limit = req.query.limit || 50;
    axios.get(`${MEMORY_CORE_URL}/diagnostics/merge_log`, { params: { limit } })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch merge log" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/config/runtime/:service", ((req: Request, res: Response) => {
    const service = req.params.service;
    axios.get(`${MEMORY_CORE_URL}/config/runtime/${service}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch runtime configuration" });
      });
  }) as RequestHandler);

  // Neural Memory routes
  apiRouter.get("/neural-memory/health", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to Neural Memory service" });
      });
  }) as RequestHandler);

  apiRouter.get("/neural-memory/status", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/status`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Neural Memory status" });
      });
  }) as RequestHandler);

  apiRouter.get("/neural-memory/diagnose_emoloop", ((req: Request, res: Response) => {
    const window = req.query.window || "24h";
    axios.get(`${NEURAL_MEMORY_URL}/diagnose_emoloop?window=${window}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch emotional loop diagnostics" });
      });
  }) as RequestHandler);

  // Context Cascade Engine routes
  apiRouter.get("/cce/health", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to CCE service" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/status", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/status`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch CCE status" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/metrics/recent_cce_responses", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/metrics/recent_cce_responses`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch recent CCE responses" });
      });
  }) as RequestHandler);

  // Configuration endpoints
  apiRouter.get("/neural-memory/config", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/config`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Neural Memory config" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/config", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/config`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch CCE config" });
      });
  }) as RequestHandler);

  // Admin action endpoints
  apiRouter.post("/memory-core/admin/verify_index", ((req: Request, res: Response) => {
    axios.post(`${MEMORY_CORE_URL}/admin/verify_index`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to trigger index verification" });
      });
  }) as RequestHandler);

  apiRouter.post("/memory-core/admin/trigger_retry_loop", ((req: Request, res: Response) => {
    axios.post(`${MEMORY_CORE_URL}/admin/trigger_retry_loop`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to trigger retry loop" });
      });
  }) as RequestHandler);

  apiRouter.post("/neural-memory/init", ((req: Request, res: Response) => {
    axios.post(`${NEURAL_MEMORY_URL}/init`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to initialize Neural Memory" });
      });
  }) as RequestHandler);

  apiRouter.post("/cce/set_variant", ((req: Request, res: Response) => {
    const { variant } = req.body;
    if (!variant) {
      return res.status(400).json({ status: "Error", message: "Variant parameter is required" });
    }
    axios.post(`${CCE_URL}/set_variant`, { variant })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to set CCE variant" });
      });
  }) as RequestHandler);

  // Alerts API (for demonstration)
  apiRouter.get("/alerts", ((req: Request, res: Response) => {
    storage.getAlerts()
      .then(alerts => {
        res.json(alerts);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch alerts" });
      });
  }) as RequestHandler);

  // Mount the router on the app
  app.use("/api", apiRouter);

  // Create the HTTP server
  const server = createServer(app);
  return server;
}

```

# server\storage.ts

```ts
import { users, type User, type InsertUser, type Alert } from "@shared/schema";

// Extend the storage interface with additional methods
export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getAlerts(): Promise<Alert[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private alerts: Alert[];
  currentId: number;

  constructor() {
    this.users = new Map();
    this.currentId = 1;
    
    // Initialize with some sample alerts
    this.alerts = [
      {
        id: "alert-1",
        type: "warning",
        title: "High gradient norm detected in Neural Memory",
        description: "The gradient norm of 0.8913 exceeds the recommended threshold of 0.7500.",
        timestamp: new Date(Date.now() - 12 * 60 * 1000).toISOString(), // 12 minutes ago
        source: "NeuralMemory"
      },
      {
        id: "alert-2",
        type: "info",
        title: "Memory Core index verification completed",
        description: "Successfully verified 342,891 memories and 6,452 assemblies. No inconsistencies found.",
        timestamp: new Date(Date.now() - 43 * 60 * 1000).toISOString(), // 43 minutes ago
        source: "MemoryCore"
      },
      {
        id: "alert-3",
        type: "warning",
        title: "CCE variant selection fluctuating",
        description: "Unusual switching between MAC-7b and MAC-13b variants detected (8 switches in 2 hours).",
        timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(), // 1 hour ago
        source: "CCE"
      }
    ];
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async getAlerts(): Promise<Alert[]> {
    return this.alerts;
  }
}

export const storage = new MemStorage();

```

# server\vite.ts

```ts
import express, { type Express } from "express";
import fs from "fs";
import path from "path";
import { createServer as createViteServer, createLogger } from "vite";
import { type Server } from "http";
import viteConfig from "../vite.config";
import { nanoid } from "nanoid";

const viteLogger = createLogger();

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

export async function setupVite(app: Express, server: Server) {
  const serverOptions = {
    middlewareMode: true,
    hmr: { server },
    allowedHosts: true as const,
  };

  const vite = await createViteServer({
    ...viteConfig,
    configFile: false,
    customLogger: {
      ...viteLogger,
      error: (msg, options) => {
        viteLogger.error(msg, options);
        process.exit(1);
      },
    },
    server: serverOptions,
    appType: "custom",
  });

  app.use(vite.middlewares);
  app.use("*", async (req, res, next) => {
    const url = req.originalUrl;

    try {
      const clientTemplate = path.resolve(
        import.meta.dirname,
        "..",
        "client",
        "index.html",
      );

      // always reload the index.html file from disk incase it changes
      let template = await fs.promises.readFile(clientTemplate, "utf-8");
      template = template.replace(
        `src="/src/main.tsx"`,
        `src="/src/main.tsx?v=${nanoid()}"`,
      );
      const page = await vite.transformIndexHtml(url, template);
      res.status(200).set({ "Content-Type": "text/html" }).end(page);
    } catch (e) {
      vite.ssrFixStacktrace(e as Error);
      next(e);
    }
  });
}

export function serveStatic(app: Express) {
  const distPath = path.resolve(import.meta.dirname, "public");

  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  app.use(express.static(distPath));

  // fall through to index.html if the file doesn't exist
  app.use("*", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}

```

# shared\schema.ts

```ts
import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Keep original user table
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Define types needed for dashboard
export interface ServiceStatus {
  name: string;
  status: 'Healthy' | 'Unhealthy' | 'Checking...' | 'Error';
  url: string;
  details?: string;
  uptime?: string;
  version?: string;
}

export interface MemoryStats {
  total_memories: number;
  total_assemblies: number;
  dirty_items: number;
  pending_vector_updates: number;
  vector_index: {
    count: number;
    mapping_count: number;
    drift_count: number;
    index_type: string;
    gpu_enabled: boolean;
  };
  assembly_stats: {
    total_count: number;
    indexed_count: number;
    vector_indexed_count: number;
    average_size: number;
    pruning_enabled: boolean;
    merging_enabled: boolean;
  };
  persistence: {
    last_update: string;
    last_backup: string;
  };
  performance: {
    quick_recall_rate: number;
    threshold_recall_rate: number;
  };
}

export interface NeuralMemoryStatus {
  initialized: boolean;
  config: {
    dimensions: number;
    hidden_size: number;
    layers: number;
  };
}

export interface NeuralMemoryDiagnostics {
  avg_loss: number;
  avg_grad_norm: number;
  avg_qr_boost: number;
  emotional_loop: {
    dominant_emotions: string[];
    entropy: number;
    bias_index: number;
    match_rate: number;
  };
  alerts: string[];
  recommendations: string[];
}

export interface CCEMetrics {
  recent_responses: CCEResponse[];
}

export interface CCEResponse {
  timestamp: string;
  status: 'success' | 'error';
  variant_output: {
    variant_type: string;
  };
  variant_selection?: {
    selected_variant: string;
    reason: string;
    performance_used: boolean;
  };
  llm_advice_used?: {
    raw_advice?: string;
    adjusted_advice: string;
    confidence_level: number;
    adjustment_reason?: string;
  };
  error_details?: string;
}

export interface Assembly {
  id: string;
  name: string;
  description: string;
  member_count: number;
  keywords: string[];
  tags: string[];
  topics: string[];
  created_at: string;
  updated_at: string;
  vector_index_updated_at?: string;
  memory_ids: string[];
}

export interface CCEConfig {
  active_variant: string;
  variant_confidence_threshold: number;
  llm_guidance_enabled: boolean;
  retry_attempts: number;
}

export interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  source: 'MemoryCore' | 'NeuralMemory' | 'CCE';
  action?: string;
}

// --- Phase 5.9 Explainability Interfaces ---

export interface ExplainActivationData {
  assembly_id: string;
  memory_id?: string | null;
  check_timestamp: string; // ISO string
  trigger_context?: string | null;
  assembly_state_before_check?: Record<string, any> | null;
  calculated_similarity?: number | null;
  activation_threshold?: number | null;
  passed_threshold?: boolean | null;
  notes?: string | null;
}

export interface ExplainActivationEmpty {
  assembly_id: string;
  memory_id?: string | null;
  notes: string;
}

export interface ExplainActivationResponse {
  success: boolean;
  explanation: ExplainActivationData | ExplainActivationEmpty;
  error?: string | null;
}

export interface ExplainMergeData {
  assembly_id: string;
  source_assembly_ids: string[];
  merge_timestamp: string;
  similarity_at_merge?: number | null;
  merge_threshold?: number | null;
  cleanup_status: 'pending' | 'completed' | 'failed';
  cleanup_timestamp?: string | null;
  cleanup_error?: string | null;
  notes?: string | null;
}

export interface ExplainMergeEmpty {
  assembly_id: string;
  notes: string;
}

export interface ExplainMergeResponse {
  success: boolean;
  explanation: ExplainMergeData | ExplainMergeEmpty;
  error?: string | null;
}

export interface LineageEntry {
  assembly_id: string;
  name?: string | null;
  depth: number;
  status?: string | null; // "origin", "merged", "cycle_detected", etc.
  created_at?: string | null; // ISO string
  memory_count?: number | null;
  parent_ids?: string[]; // IDs of source assemblies this was merged from
}

export interface LineageResponse {
  success: boolean;
  target_assembly_id: string;
  lineage: LineageEntry[];
  max_depth_reached: boolean;
  cycles_detected: boolean;
  error?: string | null;
}

// --- Phase 5.9 Diagnostics Interfaces ---

export interface ReconciledMergeLogEntry {
  merge_event_id: string;
  creation_timestamp: string; // ISO string
  source_assembly_ids: string[];
  target_assembly_id: string;
  similarity_at_merge?: number | null;
  merge_threshold?: number | null;
  final_cleanup_status: string; // "pending", "completed", "failed"
  cleanup_timestamp?: string | null; // ISO string
  cleanup_error?: string | null;
}

export interface MergeLogResponse {
  success: boolean;
  reconciled_log_entries: ReconciledMergeLogEntry[];
  count: number;
  query_limit: number;
  error?: string | null;
}

export interface RuntimeConfigResponse {
  success: boolean;
  service: string;
  config: Record<string, any>; // Sanitized config keys/values
  retrieval_timestamp: string; // ISO string
  error?: string | null;
}

export interface ActivationStats {
  total_activations: number;
  activations_by_assembly: Record<string, number>; // assembly_id -> count
  last_updated: string; // ISO timestamp
}

export interface ServiceMetrics {
  service_name: string;
  vector_operations: {
    avg_latency_ms: number;
    operation_counts: Record<string, number>; // operation -> count
  };
  persistence_operations: {
    avg_latency_ms: number;
    operation_counts: Record<string, number>; // operation -> count
  };
  // Other metrics fields
}

```

# start-dev.js

```js
#!/usr/bin/env node

import { spawn } from 'child_process';
import { createServer } from 'http';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Start the server
const serverProcess = spawn('node', ['--import=tsx', './server/index.ts'], {
  stdio: 'inherit',
  cwd: __dirname,
  env: { ...process.env, NODE_ENV: 'development' }
});

console.log('Starting Synthians Cognitive Dashboard development server...');

serverProcess.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
  process.exit(code);
});

process.on('SIGINT', () => {
  console.log('Shutting down development server...');
  serverProcess.kill('SIGINT');
});

process.on('SIGTERM', () => {
  console.log('Shutting down development server...');
  serverProcess.kill('SIGTERM');
});

```

# tailwind.config.ts

```ts
import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./client/index.html", "./client/src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate"), require("@tailwindcss/typography")],
} satisfies Config;

```

# theme.json

```json
{
  "variant": "professional",
  "primary": "#FF008C",
  "appearance": "dark",
  "radius": 0.5
}

```

# tsconfig.json

```json
{
  "include": ["client/src/**/*", "shared/**/*", "server/**/*"],
  "exclude": ["node_modules", "build", "dist", "**/*.test.ts"],
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": "./node_modules/typescript/tsbuildinfo",
    "noEmit": true,
    "module": "ESNext",
    "strict": true,
    "lib": ["esnext", "dom", "dom.iterable"],
    "jsx": "preserve",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "allowImportingTsExtensions": true,
    "moduleResolution": "bundler",
    "baseUrl": ".",
    "types": ["node", "vite/client"],
    "paths": {
      "@/*": ["./client/src/*"],
      "@shared/*": ["./shared/*"]
    }
  }
}

```

# vite.config.ts

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import themePlugin from "@replit/vite-plugin-shadcn-theme-json";
import path from "path";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";

// Use path.resolve directly based on __dirname
const __dirname = path.resolve();

export default defineConfig({
  plugins: [
    // Use React plugin without extra JSX options - let it handle automatic JSX runtime
    react(),
    runtimeErrorOverlay(),
    themePlugin(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "client", "src"),
      "@shared": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "shared"),
      "@assets": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "attached_assets"),
    },
  },
  root: path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "client"),
  build: {
    outDir: path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "dist", "public"),
    emptyOutDir: true,
  },
});

```

