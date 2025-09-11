from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent

from autogpt.core.resource.model_providers import ChatMessage


class AgentContext:
    """Manages contextual information for agent interactions.
    
    This class maintains a collection of context items that provide additional
    information to help the agent understand and respond to tasks. Context items
    can include file contents, previous conversation history, external data, or
    any other relevant information.
    
    The context is designed to be dynamic, allowing items to be added, removed,
    and formatted for inclusion in agent prompts.
    
    Attributes:
        items: List of ContextItem objects containing contextual information
    """
    items: list[ContextItem]

    def __init__(self, items: Optional[list[ContextItem]] = None):
        """Initialize the agent context.
        
        Args:
            items: Optional initial list of context items. If None, starts with empty context.
        """
        self.items = items or []
=======

    def __bool__(self) -> bool:
        """Check if the context contains any items.
        
        Returns:
            bool: True if there are context items, False if empty
        """
        return len(self.items) > 0
=======

    def __contains__(self, item: ContextItem) -> bool:
        """Check if a context item with the same source already exists.
        
        Args:
            item: The context item to check for
            
        Returns:
            bool: True if an item with the same source exists, False otherwise
        """
        return any([i.source == item.source for i in self.items])
=======

    def add(self, item: ContextItem) -> None:
        """Add a new context item to the collection.
        
        Args:
            item: The context item to add
        """
        self.items.append(item)
=======

    def close(self, index: int) -> None:
        """Remove a context item by its 1-based index.
        
        This method is typically used when a context item is no longer needed
        and should be removed from the agent's working context.
        
        Args:
            index: 1-based index of the item to remove (as displayed to users)
            
        Raises:
            IndexError: If the index is out of range
        """
        self.items.pop(index - 1)
=======

    def clear(self) -> None:
        """Remove all context items from the collection."""
        self.items.clear()
=======

    def format_numbered(self) -> str:
        return "\n\n".join([f"{i}. {c.fmt()}" for i, c in enumerate(self.items, 1)])


class ContextMixin:
    """Mixin that adds context support to a BaseAgent subclass"""

    context: AgentContext

    def __init__(self, **kwargs: Any):
        self.context = AgentContext()

        super(ContextMixin, self).__init__(**kwargs)

    def build_prompt(
        self,
        *args: Any,
        extra_messages: Optional[list[ChatMessage]] = None,
        **kwargs: Any,
    ) -> ChatPrompt:
        if not extra_messages:
            extra_messages = []

        # Add context section to prompt
        if self.context:
            extra_messages.insert(
                0,
                ChatMessage.system(
                    "## Context\n"
                    f"{self.context.format_numbered()}\n\n"
                    "When a context item is no longer needed and you are not done yet, "
                    "you can hide the item by specifying its number in the list above "
                    "to `hide_context_item`.",
                ),
            )

        return super(ContextMixin, self).build_prompt(
            *args,
            extra_messages=extra_messages,
            **kwargs,
        )  # type: ignore


def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    if isinstance(agent, ContextMixin):
        return agent.context

    return None
