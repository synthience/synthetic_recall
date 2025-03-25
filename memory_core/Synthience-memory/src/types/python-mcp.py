from typing import Dict, Any, List, Callable, Optional

class ErrorCode:
    METHOD_NOT_FOUND = "MethodNotFound"
    INVALID_REQUEST = "InvalidRequest"
    INTERNAL_ERROR = "InternalError"

class McpError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

# Request schema identifiers
LIST_TOOLS_REQUEST_SCHEMA = "list_tools"
CALL_TOOL_REQUEST_SCHEMA = "call_tool"

class StdioServerTransport:
    """Simple stdio-based transport for the MCP server."""
    def __init__(self):
        self.is_connected = False
    
    async def connect(self, message_handler):
        """Set up the connection."""
        self.is_connected = True
        self.message_handler = message_handler
    
    async def send(self, message: Dict[str, Any]):
        """Send a message through stdio."""
        print(f"MCP >>> {message}")
    
    async def close(self):
        """Close the transport."""
        self.is_connected = False

class Server:
    """Python implementation of the MCP server."""
    def __init__(self, info: Dict[str, str], config: Dict[str, Any]):
        self.info = info
        self.config = config
        self.handlers = {}
        self.transport = None
        self.onerror = None
    
    def set_request_handler(self, schema: str, handler: Callable):
        """Register a handler for a specific request schema."""
        self.handlers[schema] = handler
    
    async def connect(self, transport):
        """Connect the server to a transport."""
        self.transport = transport
        await self.transport.connect(self.handle_message)
    
    async def handle_message(self, message):
        """Process incoming messages."""
        try:
            schema = message.get("schema")
            if schema in self.handlers:
                response = await self.handlers[schema](message)
                await self.transport.send(response)
            else:
                error = McpError(ErrorCode.METHOD_NOT_FOUND, f"Unknown schema: {schema}")
                if self.onerror:
                    self.onerror(error)
                await self.transport.send({
                    "error": {
                        "code": error.code,
                        "message": error.message
                    }
                })
        except Exception as error:
            if self.onerror:
                self.onerror(error)
            await self.transport.send({
                "error": {
                    "code": getattr(error, "code", ErrorCode.INTERNAL_ERROR),
                    "message": str(error)
                }
            })
    
    async def close(self):
        """Close the server."""
        if self.transport:
            await self.transport.close()
