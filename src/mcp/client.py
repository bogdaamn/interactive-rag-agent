import json
import subprocess
import threading
from pathlib import Path

class MCPClient:
    """Client for communicating with MCP server."""
    
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        self.lock = threading.Lock()
        self.next_id = 1
        self.initialize()

    def _send(self, payload):
        """Send JSON-RPC message and receive response."""
        with self.lock:
            data = json.dumps(payload)
            self.proc.stdin.write(data + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline()
            if not line:
                raise ConnectionError("MCP server disconnected")
            return json.loads(line)

    def initialize(self):
        """Initialize MCP connection."""
        payload = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "company-kb-assistant", "version": "1.0"}
            }
        }
        self.next_id += 1

        return self._send(payload)

    def call_tool(self, name, arguments):
        """Call an MCP tool."""
        payload = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        self.next_id += 1

        return self._send(payload)
    
    def close(self):
        """Close the MCP connection."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
