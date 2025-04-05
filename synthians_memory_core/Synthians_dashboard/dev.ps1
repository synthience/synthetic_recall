# Set environment variables for development
$env:NODE_ENV = "development"

# Run the server with the proper import flag
node --import tsx/esm server/index.ts
