"""LLM Agent with MCP tool access via OpenRouter API."""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMAgent:
    """AI agent that can call MCP tools through MCP Host."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.mcp_executor: Optional[Callable] = None
        self.call_count = 0
        self.start_time = 0.0

    def set_mcp_executor(self, executor: Callable) -> None:
        """Set callback for executing MCP tool calls."""
        self.mcp_executor = executor

    def analyze_with_mcp(
        self, task: str, available_tools: List[Dict], max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run analysis with iterative MCP tool calls."""
        self.start_time = time.time()
        self.call_count = 0

        system_prompt = self._build_tools_prompt(available_tools)
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for iteration in range(max_iterations):
            response = self._call_llm(conversation)
            content = response.get("content", "")

            # Check if LLM wants to call a tool
            tool_call = self._parse_tool_call(content)
            if tool_call and self.mcp_executor:
                tool_result = self.mcp_executor(tool_call["tool"], tool_call["args"])
                self.call_count += 1
                conversation.append({"role": "assistant", "content": content})
                conversation.append(
                    {
                        "role": "user",
                        "content": f"Tool '{tool_call['tool']}' result: {json.dumps(tool_result, ensure_ascii=False)}",
                    }
                )
            else:
                # Final result
                elapsed = time.time() - self.start_time
                return {
                    "content": content,
                    "iterations": iteration + 1,
                    "mcp_calls": self.call_count,
                    "time_seconds": round(elapsed, 2),
                }

        elapsed = time.time() - self.start_time
        return {
            "content": conversation[-1].get("content", ""),
            "iterations": max_iterations,
            "mcp_calls": self.call_count,
            "time_seconds": round(elapsed, 2),
        }

    def _build_tools_prompt(self, tools: List[Dict]) -> str:
        """Build system prompt with available MCP tools."""
        tools_desc = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            tools_desc.append(f"- {name}: {desc}")

        return f"""Ты анализируешь GitHub репозиторий через MCP инструменты.

Доступные инструменты:
{chr(10).join(tools_desc)}

Для вызова инструмента верни JSON:
{{"action": "call_tool", "tool": "tool_name", "args": {{"arg1": "value1"}}}}

Для финального результата верни JSON:
{{"action": "result", "metrics": {{"metric1": value1}}}}

Всегда возвращай валидный JSON."""

    def _call_llm(self, messages: List[Dict]) -> Dict[str, Any]:
        """Call OpenRouter API."""
        resp = requests.post(
            OPENROUTER_URL,
            json={"model": self.model, "messages": messages},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/elenagernichenko/github-analyzer",
                "X-Title": "github-analyzer",
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from LLM response."""
        content = content.strip()
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(content)
            if data.get("action") == "call_tool":
                return {"tool": data.get("tool"), "args": data.get("args", {})}
        except (json.JSONDecodeError, KeyError):
            pass
        return None

