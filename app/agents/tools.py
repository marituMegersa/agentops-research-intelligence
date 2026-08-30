"""Dynamic Tool Registry, Tool Definition schemas, and Execution Dispatcher."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Schema for a single tool argument."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Metadata and execution pointer for a registered agent tool."""

    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ToolRegistry:
    """Central registry managing dynamic tool registration, discovery, and execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._executables: Dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Decorator to register a Python function as a tool."""

        def decorator(func: Callable[..., Any]):
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()

            params: List[ToolParameter] = []
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                type_name = getattr(param.annotation, "__name__", str(param.annotation))
                required = param.default is inspect.Parameter.empty
                default_val = None if required else param.default
                params.append(
                    ToolParameter(
                        name=param_name,
                        type=type_name,
                        required=required,
                        default=default_val,
                    )
                )

            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=params,
                tags=tags or [],
            )
            self._tools[tool_name] = tool_def
            self._executables[tool_name] = func
            return func

        return decorator

    def register_tool(
        self,
        tool_def: ToolDefinition,
        func: Callable[..., Any],
    ) -> None:
        """Explicitly register a ToolDefinition and its callable."""
        self._tools[tool_def.name] = tool_def
        self._executables[tool_def.name] = func

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve tool metadata by name."""
        return self._tools.get(name)

    def list_tools(self, tag: Optional[str] = None) -> List[ToolDefinition]:
        """List all registered tools, optionally filtered by tag."""
        if tag:
            return [t for t in self._tools.values() if tag in t.tags]
        return list(self._tools.values())

    def execute(self, name: str, **kwargs) -> Any:
        """Dispatch and execute a registered tool with provided kwargs."""
        if name not in self._executables:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        func = self._executables[name]
        return func(**kwargs)

    def select_tools_for_query(self, query: str, limit: int = 3) -> List[ToolDefinition]:
        """Match query intent to relevant tools based on tags and description overlap."""
        query_words = set(query.lower().split())
        scored_tools = []

        for tool in self._tools.values():
            text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            score = sum(1 for w in query_words if w in text)
            scored_tools.append((score, tool))

        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool for score, tool in scored_tools[:limit]]
