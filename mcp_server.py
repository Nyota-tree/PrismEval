"""
PrismEval MCP Server

Exposes the `evaluate` tool via MCP stdio transport so Claude Code,
OpenClaw, and other MCP-compatible agents can call PrismEval's
quality-judge pipeline programmatically.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure configs/ is always found regardless of the working directory
os.chdir(Path(__file__).parent)

# Make the project root importable when the file is run directly
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from prism_eval.core.pipeline import fill_evaluation_prompt
from prism_eval.metrics.extractor import extract_json_from_text, extract_evaluation
from prism_eval.providers.llm_client import LLMClient
from prism_eval.utils.config_loader import get_evaluation_prompt_template

app = Server("prism-eval")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="evaluate",
            description=(
                "Evaluate an LLM-generated output using PrismEval's quality-judge pipeline. "
                "Returns structured scores (factuality, completeness, north_star, weighted_total) "
                "and an automated PUBLISH / REVIEW / REJECT decision."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The original input, question, or prompt given to the LLM.",
                    },
                    "output": {
                        "type": "string",
                        "description": "The LLM-generated output to evaluate.",
                    },
                    "criteria": {
                        "type": "string",
                        "description": (
                            "Optional custom evaluation criteria appended to the default "
                            "prompt template. Use this to add domain-specific scoring guidance."
                        ),
                    },
                },
                "required": ["input", "output"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "evaluate":
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )
        ]

    input_text: str = arguments.get("input", "")
    output_text: str = arguments.get("output", "")
    criteria: Optional[str] = arguments.get("criteria", "")

    result = await _run_evaluation(input_text, output_text, criteria)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2),
        )
    ]


async def _run_evaluation(
    input_text: str,
    output_text: str,
    criteria: Optional[str],
) -> Dict[str, Any]:
    """Run the full evaluation pipeline asynchronously."""
    # 1. Load the default evaluation prompt template
    template = get_evaluation_prompt_template()
    if not template:
        return {"error": "Evaluation prompt template is empty. Check configs/prompts.yaml."}

    # 2. Fill {original_text} / {model_output} placeholders
    filled_prompt = fill_evaluation_prompt(template, input_text, output_text)

    # 3. Append custom criteria when provided
    if criteria and criteria.strip():
        filled_prompt = filled_prompt + "\n\n" + criteria.strip()

    # 4. Call the LLM (synchronous) via executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        client = LLMClient()  # provider/model/temperature come from configs/default.yaml
        raw_response: str = await loop.run_in_executor(
            None, lambda: client.generate(system_prompt="", user_prompt=filled_prompt)
        )
    except Exception as exc:
        return {"error": f"LLM call failed: {exc}", "raw": ""}

    # 5. Extract JSON from the raw LLM response
    json_obj = extract_json_from_text(raw_response)
    if json_obj is None:
        return {
            "error": "Could not extract JSON from LLM response.",
            "raw": raw_response[:500],
        }

    # 6. Normalize scores and apply PUBLISH / REVIEW / REJECT decision rules
    try:
        return extract_evaluation(json_obj)
    except Exception as exc:
        return {
            "error": f"Score extraction failed: {exc}",
            "raw": raw_response[:500],
        }


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
