"""Chat screen — interactive conversation with the loaded model."""

import threading
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Input, Button, Label
from textual.widget import Widget

from llm_studio.models.engine import ChatMessage
from llm_studio.ui.widgets.message_list import ChatMessageList, StreamingBubble


class ChatScreen(Widget):
    """Interactive chat screen with the loaded LLM."""

    DEFAULT_CSS = """
    ChatScreen {
        width: 1fr;
        height: 1fr;
    }
    #chat-header {
        height: 3;
        background: $surface;
        border-bottom: tall $primary-background;
        content-align: center middle;
        padding: 0 2;
    }
    #chat-input-bar {
        height: 3;
        dock: bottom;
        background: $surface;
        border-top: tall $primary-background;
        padding: 0 1;
    }
    #chat-input {
        width: 1fr;
    }
    #btn-send {
        width: 12;
        margin: 0 0 0 1;
    }
    #btn-clear {
        width: 10;
        margin: 0 0 0 1;
    }
    .no-model-warning {
        width: 100%;
        height: 100%;
        content-align: center middle;
        text-align: center;
        color: $warning;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chat_history: list[ChatMessage] = []
        self._is_generating = False
        self._current_bubble: StreamingBubble | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            " 💬 [bold]Chat[/]  [dim]│  Ctrl+Enter to send  │  Esc to cancel[/]",
            id="chat-header",
        )
        yield ChatMessageList(id="chat-messages")
        with Horizontal(id="chat-input-bar"):
            yield Input(
                placeholder="Type your message here...",
                id="chat-input",
            )
            yield Button("Send ➤", id="btn-send", variant="primary")
            yield Button("Clear", id="btn-clear", variant="error")

    def on_mount(self):
        """Add initial system message."""
        if not self._chat_history:
            sys_prompt = self.app.config.system_prompt
            self._chat_history.append(
                ChatMessage(role="system", content=sys_prompt)
            )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-send":
            self._send_message()
        elif event.button.id == "btn-clear":
            self._clear_chat()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "chat-input":
            self._send_message()

    def _send_message(self):
        """Send user message and start generation."""
        if self._is_generating:
            return

        engine = self.app.engine
        if not engine.is_loaded:
            self.app.notify(
                "No model loaded! Go to Models tab first.",
                severity="warning",
            )
            return

        chat_input = self.query_one("#chat-input", Input)
        text = chat_input.value.strip()
        if not text:
            return

        chat_input.value = ""

        # Add user message
        user_msg = ChatMessage(role="user", content=text)
        self._chat_history.append(user_msg)
        msg_list = self.query_one("#chat-messages", ChatMessageList)
        msg_list.add_message("user", text)

        # Start generation in background thread
        self._is_generating = True
        self._current_bubble = msg_list.add_streaming_bubble()

        thread = threading.Thread(
            target=self._generate_response,
            daemon=True,
        )
        thread.start()

    def _generate_response(self):
        """Generate assistant response (runs in background thread)."""
        engine = self.app.engine
        full_response = ""

        try:
            for token in engine.chat_completion_stream(self._chat_history):
                full_response += token
                if self._current_bubble:
                    self.app.call_from_thread(
                        self._current_bubble.append_token, token
                    )
        except Exception as e:
            full_response = f"[Error] {e}"
            if self._current_bubble:
                self.app.call_from_thread(
                    self._current_bubble.append_token,
                    f"\n\n[red][Error: {e}][/red]"
                )

        # Finalize
        if self._current_bubble:
            self.app.call_from_thread(self._current_bubble.finalize)

        self._chat_history.append(
            ChatMessage(role="assistant", content=full_response)
        )
        self._is_generating = False
        self._current_bubble = None

        # Scroll to end
        try:
            msg_list = self.query_one("#chat-messages", ChatMessageList)
            self.app.call_from_thread(msg_list.scroll_end, animate=False)
        except Exception:
            pass

    def _clear_chat(self):
        """Clear chat history and UI."""
        self._chat_history.clear()
        sys_prompt = self.app.config.system_prompt
        self._chat_history.append(
            ChatMessage(role="system", content=sys_prompt)
        )
        msg_list = self.query_one("#chat-messages", ChatMessageList)
        msg_list.clear_messages()
        self.app.notify("Chat cleared", severity="information")
