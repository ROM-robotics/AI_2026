"""Home screen — dashboard overview."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Button, Label
from textual.widget import Widget

import shutil
from pathlib import Path


class InfoCard(Static):
    """A dashboard info card."""

    DEFAULT_CSS = """
    InfoCard {
        width: 1fr;
        height: auto;
        min-height: 7;
        margin: 1;
        padding: 1 2;
        background: $surface;
        border: round $primary-background;
    }
    """


class HomeScreen(Widget):
    """Main home/dashboard screen."""

    DEFAULT_CSS = """
    HomeScreen {
        width: 1fr;
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="home-content"):
            yield Static(
                "\n"
                "  [bold cyan]╔══════════════════════════════════════════════════╗[/]\n"
                "  [bold cyan]║[/]     [bold white]◆  Welcome to LLM Studio  ◆[/]          [bold cyan]║[/]\n"
                "  [bold cyan]║[/]     [dim]Your Local LLM Playground[/]               [bold cyan]║[/]\n"
                "  [bold cyan]╚══════════════════════════════════════════════════╝[/]\n",
                id="home-banner",
            )

            with Horizontal(id="home-cards"):
                yield InfoCard(
                    self._model_info(),
                    id="card-models",
                )
                yield InfoCard(
                    self._server_info(),
                    id="card-server",
                )
                yield InfoCard(
                    self._system_info(),
                    id="card-system",
                )

            yield Static("─" * 60, classes="home-divider")
            yield Static(
                "\n  [bold]Quick Actions[/]\n",
                classes="home-section-title",
            )
            with Horizontal(id="home-actions"):
                yield Button("💬  Start Chat", id="btn-go-chat", variant="primary")
                yield Button("📦  Manage Models", id="btn-go-models", variant="default")
                yield Button("🌐  Start Server", id="btn-go-server", variant="success")
                yield Button("⚙️   Settings", id="btn-go-settings", variant="default")

            yield Static("─" * 60, classes="home-divider")
            yield Static(
                "\n  [bold]Getting Started[/]\n\n"
                "  [dim]1.[/]  Go to [bold]Models[/] tab to download a GGUF model from HuggingFace\n"
                "  [dim]2.[/]  Load the model from the model list\n"
                "  [dim]3.[/]  Go to [bold]Chat[/] tab to start chatting\n"
                "  [dim]4.[/]  Optionally start the [bold]Server[/] for OpenAI-compatible API access\n\n"
                "  [dim]Supported formats:[/] GGUF (llama.cpp)\n"
                "  [dim]API compatible:[/]    OpenAI Chat Completions, Completions, Embeddings\n",
                id="home-guide",
            )

    def _model_info(self) -> str:
        """Get model info for the dashboard card."""
        app = self.app
        models = []
        try:
            models = app.model_manager.list_local_models()
        except Exception:
            pass

        loaded = "None"
        if hasattr(app, 'engine') and app.engine.is_loaded:
            loaded = Path(app.engine.model_path).stem

        return (
            " [bold]📦 Models[/]\n"
            f" ─────────────────\n"
            f" Installed: [bold]{len(models)}[/]\n"
            f" Loaded:    [bold cyan]{loaded}[/]\n"
        )

    def _server_info(self) -> str:
        """Get server info for the dashboard card."""
        app = self.app
        running = False
        url = "N/A"
        try:
            running = app.server.is_running
            url = app.server.base_url
        except Exception:
            pass

        status = "[bold green]● RUNNING[/]" if running else "[bold red]● STOPPED[/]"
        return (
            " [bold]🌐 Server[/]\n"
            f" ─────────────────\n"
            f" Status:  {status}\n"
            f" URL:     [bold]{url}[/]\n"
        )

    def _system_info(self) -> str:
        """Get system info for the dashboard card."""
        try:
            disk = shutil.disk_usage(str(Path.home()))
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
        except Exception:
            free_gb = 0
            total_gb = 0

        import os
        cpu_count = os.cpu_count() or 0

        return (
            " [bold]🖥️  System[/]\n"
            f" ─────────────────\n"
            f" CPU Cores: [bold]{cpu_count}[/]\n"
            f" Disk Free: [bold]{free_gb:.1f}[/] / {total_gb:.1f} GB\n"
        )

    def refresh_cards(self):
        """Refresh dashboard cards with current info."""
        try:
            self.query_one("#card-models", InfoCard).update(self._model_info())
            self.query_one("#card-server", InfoCard).update(self._server_info())
            self.query_one("#card-system", InfoCard).update(self._system_info())
        except Exception:
            pass
