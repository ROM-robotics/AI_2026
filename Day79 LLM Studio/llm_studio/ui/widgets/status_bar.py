"""Status bar widget displayed at bottom of the application."""

from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from textual.app import ComposeResult
from textual.containers import Horizontal


class StatusBar(Widget):
    """A status bar showing model info, server status, and system info."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary-background;
        color: $text-muted;
    }
    StatusBar Horizontal {
        width: 100%;
        height: 1;
    }
    StatusBar .status-left {
        width: 1fr;
        padding: 0 1;
        content-align: left middle;
    }
    StatusBar .status-center {
        width: auto;
        padding: 0 2;
        content-align: center middle;
    }
    StatusBar .status-right {
        width: auto;
        min-width: 30;
        padding: 0 1;
        content-align: right middle;
    }
    """

    model_name: reactive[str] = reactive("No model loaded")
    server_status: reactive[str] = reactive("Server: OFF")
    info_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(self.model_name, classes="status-left", id="sb-model")
            yield Static(self.server_status, classes="status-center", id="sb-server")
            yield Static(self.info_text, classes="status-right", id="sb-info")

    def watch_model_name(self, value: str):
        try:
            self.query_one("#sb-model", Static).update(f" ◆ {value}")
        except Exception:
            pass

    def watch_server_status(self, value: str):
        try:
            self.query_one("#sb-server", Static).update(value)
        except Exception:
            pass

    def watch_info_text(self, value: str):
        try:
            self.query_one("#sb-info", Static).update(value)
        except Exception:
            pass
