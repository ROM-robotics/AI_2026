"""Chat message display widgets."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, RichLog
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive

from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel


class MessageBubble(Static):
    """A single chat message bubble."""

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        margin: 1 2;
        padding: 1 2;
    }
    MessageBubble.-user {
        background: $primary 20%;
        border-left: thick $primary;
    }
    MessageBubble.-assistant {
        background: $secondary 15%;
        border-left: thick $success;
    }
    MessageBubble.-system {
        background: $warning 10%;
        border-left: thick $warning;
        text-style: italic;
    }
    """

    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self.content = content

        if role == "user":
            prefix = "[bold cyan]⟩ You[/bold cyan]\n"
            cls = "-user"
        elif role == "assistant":
            prefix = "[bold green]◆ Assistant[/bold green]\n"
            cls = "-assistant"
        else:
            prefix = "[bold yellow]⚙ System[/bold yellow]\n"
            cls = "-system"

        super().__init__(prefix + content, **kwargs)
        self.add_class(cls)


class StreamingBubble(Static):
    """A chat bubble that supports streaming content updates."""

    DEFAULT_CSS = """
    StreamingBubble {
        width: 100%;
        margin: 1 2;
        padding: 1 2;
        background: $secondary 15%;
        border-left: thick $success;
    }
    """

    content_text: reactive[str] = reactive("")

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._prefix = "[bold green]◆ Assistant[/bold green]\n"

    def watch_content_text(self, value: str):
        self.update(self._prefix + value + " [blink]▊[/blink]")

    def append_token(self, token: str):
        self.content_text += token

    def finalize(self):
        """Remove the cursor and finalize the message."""
        self.update(self._prefix + self.content_text)


class ChatMessageList(VerticalScroll):
    """Scrollable list of chat messages."""

    DEFAULT_CSS = """
    ChatMessageList {
        width: 100%;
        height: 1fr;
        background: $surface;
    }
    """

    def add_message(self, role: str, content: str):
        """Add a completed message to the list."""
        bubble = MessageBubble(role, content)
        self.mount(bubble)
        self.scroll_end(animate=False)

    def add_streaming_bubble(self) -> StreamingBubble:
        """Add a streaming bubble and return it for token appending."""
        bubble = StreamingBubble()
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def clear_messages(self):
        """Remove all messages."""
        for child in self.query(MessageBubble):
            child.remove()
        for child in self.query(StreamingBubble):
            child.remove()
