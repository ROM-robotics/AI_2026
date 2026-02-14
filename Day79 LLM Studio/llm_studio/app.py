"""
LLM Studio — Main TUI Application

A terminal-based LLM management studio powered by:
  - Textual (TUI framework)
  - llama-cpp-python (inference engine)
  - FastAPI (OpenAI-compatible API server)
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Header, Footer

from llm_studio.config import AppConfig
from llm_studio.models.manager import ModelManager
from llm_studio.models.engine import InferenceEngine
from llm_studio.server.api import LLMServer

from llm_studio.ui.widgets.sidebar import Sidebar, SidebarItem
from llm_studio.ui.widgets.status_bar import StatusBar

from llm_studio.ui.screens.home import HomeScreen
from llm_studio.ui.screens.chat import ChatScreen
from llm_studio.ui.screens.models import ModelsScreen
from llm_studio.ui.screens.server import ServerScreen
from llm_studio.ui.screens.settings import SettingsScreen


class LLMStudioApp(App):
    """LLM Studio — Your Local LLM Playground."""

    TITLE = "LLM Studio"
    SUB_TITLE = "Local LLM Playground"

    CSS_PATH = "ui/styles/app.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+t", "toggle_dark", "Theme", show=True),
        Binding("f1", "switch_screen('home')", "Home", show=True),
        Binding("f2", "switch_screen('chat')", "Chat", show=True),
        Binding("f3", "switch_screen('models')", "Models", show=True),
        Binding("f4", "switch_screen('server')", "Server", show=True),
        Binding("f5", "switch_screen('settings')", "Settings", show=True),
    ]

    def __init__(self):
        super().__init__()
        # ── Core services ──────────────────────────────────────────
        self.config = AppConfig.load()
        self.model_manager = ModelManager(self.config)
        self.engine = InferenceEngine(self.config.inference)
        self.server = LLMServer(self.engine, self.config.server)

        # ── Screen registry ────────────────────────────────────────
        self._screens: dict[str, type] = {
            "home": HomeScreen,
            "chat": ChatScreen,
            "models": ModelsScreen,
            "server": ServerScreen,
            "settings": SettingsScreen,
        }
        self._current_screen_name = "home"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            yield Sidebar(id="sidebar")
            with Vertical(id="screen-container"):
                yield HomeScreen(id="screen-home")
                yield ChatScreen(id="screen-chat")
                yield ModelsScreen(id="screen-models")
                yield ServerScreen(id="screen-server")
                yield SettingsScreen(id="screen-settings")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self):
        """Initialize the app — show home screen by default."""
        self._show_screen("home")
        self._update_status_bar()

    def on_button_pressed(self, event):
        """Handle navigation button presses."""
        btn_id = getattr(event.button, "id", "")

        # Sidebar navigation
        if isinstance(event.button, SidebarItem):
            self.action_switch_screen(event.button.screen_name)
            return

        # Home screen quick action buttons
        nav_map = {
            "btn-go-chat": "chat",
            "btn-go-models": "models",
            "btn-go-server": "server",
            "btn-go-settings": "settings",
        }
        if btn_id in nav_map:
            self.action_switch_screen(nav_map[btn_id])

    def action_switch_screen(self, screen_name: str):
        """Switch the active screen panel."""
        self._show_screen(screen_name)

    def _show_screen(self, name: str):
        """Show the specified screen and hide others."""
        self._current_screen_name = name

        for sn in self._screens:
            try:
                widget = self.query_one(f"#screen-{sn}")
                widget.display = (sn == name)
            except Exception:
                pass

        # Update sidebar highlight
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.active_screen = name
        except Exception:
            pass

        # Refresh home cards when switching to home
        if name == "home":
            try:
                home = self.query_one("#screen-home", HomeScreen)
                home.refresh_cards()
            except Exception:
                pass

    def _update_status_bar(self):
        """Update the status bar with current state."""
        try:
            sb = self.query_one("#status-bar", StatusBar)
            if self.engine.is_loaded:
                from pathlib import Path
                sb.model_name = Path(self.engine.model_path).stem
            else:
                sb.model_name = "No model loaded"

            if self.server.is_running:
                sb.server_status = f"[green]Server: ON ({self.server.base_url})[/]"
            else:
                sb.server_status = "[red]Server: OFF[/]"
        except Exception:
            pass

    def action_toggle_dark(self):
        """Toggle dark/light theme."""
        self.dark = not self.dark

    def action_quit(self):
        """Clean up and quit."""
        try:
            if self.server.is_running:
                self.server.stop()
            if self.engine.is_loaded:
                self.engine.unload_model()
            self.config.save()
        except Exception:
            pass
        self.exit()


def main():
    """Entry point for LLM Studio."""
    app = LLMStudioApp()
    app.run()


if __name__ == "__main__":
    main()
