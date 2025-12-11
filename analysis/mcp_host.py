"""MCP Host - orchestrates LLM agent and MCP Client."""

import json
from typing import Any, Dict, List

from mcp_session import MCPDockerSession, extract_text_content


class MCPHost:
    """MCP Host managing LLM agent lifecycle and MCP Server communication."""

    def __init__(self, mcp_session: MCPDockerSession):
        self.mcp_session = mcp_session
        self.available_tools: List[Dict] = []

    def initialize(self) -> None:
        """Initialize MCP connection and fetch available tools."""
        tools_response = self.mcp_session.list_tools()
        # tools/list returns result.tools directly, not in content
        result = tools_response.get("result", {})
        if "tools" in result:
            self.available_tools = result["tools"]
        else:
            # Fallback: try to extract from content
            tools_data = extract_text_content(tools_response)
            if tools_data:
                try:
                    tools_list = json.loads(tools_data)
                    self.available_tools = tools_list if isinstance(tools_list, list) else tools_list.get("tools", [])
                except json.JSONDecodeError:
                    self.available_tools = []
            else:
                self.available_tools = []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute MCP tool call and return parsed result."""
        response = self.mcp_session.call_tool(tool_name, arguments)
        content = extract_text_content(response)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

