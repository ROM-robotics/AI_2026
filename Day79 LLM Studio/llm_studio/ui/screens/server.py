"""Server screen — manage the OpenAI-compatible API server."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Input, Label, Log
from textual.widget import Widget


class ServerScreen(Widget):
    """API server management screen."""

    DEFAULT_CSS = """
    ServerScreen {
        width: 1fr;
        height: 1fr;
    }
    #server-header {
        height: 3;
        background: $surface;
        border-bottom: tall $primary-background;
        padding: 0 2;
        content-align: left middle;
    }
    #server-content {
        padding: 1 2;
    }
    #server-status-panel {
        height: auto;
        min-height: 10;
        padding: 1 2;
        margin: 1 0;
        background: $surface;
        border: round $primary-background;
    }
    #server-config-panel {
        height: auto;
        padding: 1 2;
        margin: 1 0;
        background: $surface;
        border: round $primary-background;
    }
    #server-endpoints {
        height: auto;
        padding: 1 2;
        margin: 1 0;
        background: $surface;
        border: round $primary-background;
    }
    .server-buttons {
        height: 3;
        margin: 1 0;
    }
    .server-buttons Button {
        margin: 0 1 0 0;
    }
    .config-row {
        height: 3;
        margin: 0 0 1 0;
    }
    .config-label {
        width: 14;
        content-align: right middle;
        padding: 0 1;
    }
    .config-input {
        width: 1fr;
    }
    #server-log {
        height: 1fr;
        min-height: 8;
        margin: 1 0;
        border: round $primary-background;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            " 🌐 [bold]API Server[/]  [dim]│  OpenAI-compatible REST API[/]",
            id="server-header",
        )
        with Vertical(id="server-content"):
            # ── Status panel ───────────────────────────────────────
            with Vertical(id="server-status-panel"):
                yield Static("[bold]Server Status[/]\n", classes="section-title")
                yield Static(self._status_text(), id="server-status-text")
                with Horizontal(classes="server-buttons"):
                    yield Button(
                        "▶  Start Server",
                        id="btn-start-server",
                        variant="success",
                    )
                    yield Button(
                        "⏹  Stop Server",
                        id="btn-stop-server",
                        variant="error",
                    )
                    yield Button(
                        "🔄  Restart",
                        id="btn-restart-server",
                        variant="warning",
                    )

            # ── Configuration panel ────────────────────────────────
            with Vertical(id="server-config-panel"):
                yield Static("[bold]Configuration[/]\n")
                with Horizontal(classes="config-row"):
                    yield Static("Host:", classes="config-label")
                    yield Input(
                        value=self.app.config.server.host,
                        id="server-host-input",
                        classes="config-input",
                    )
                with Horizontal(classes="config-row"):
                    yield Static("Port:", classes="config-label")
                    yield Input(
                        value=str(self.app.config.server.port),
                        id="server-port-input",
                        classes="config-input",
                    )
                with Horizontal(classes="config-row"):
                    yield Static("API Key:", classes="config-label")
                    yield Input(
                        value=self.app.config.server.api_key or "",
                        placeholder="Leave empty for no auth",
                        id="server-apikey-input",
                        classes="config-input",
                        password=True,
                    )

            # ── Endpoints info ─────────────────────────────────────
            with Vertical(id="server-endpoints"):
                yield Static(self._endpoints_text(), id="endpoints-info")

    def _status_text(self) -> str:
        """Generate server status text."""
        try:
            running = self.app.server.is_running
            url = self.app.server.base_url
        except Exception:
            running = False
            url = "N/A"

        model_loaded = False
        model_name = "None"
        try:
            model_loaded = self.app.engine.is_loaded
            if model_loaded:
                from pathlib import Path
                model_name = Path(self.app.engine.model_path).stem
        except Exception:
            pass

        if running:
            status = "[bold green]● RUNNING[/bold green]"
        else:
            status = "[bold red]● STOPPED[/bold red]"

        return (
            f" Status:     {status}\n"
            f" URL:        [bold]{url}[/bold]\n"
            f" Model:      [bold cyan]{model_name}[/bold cyan]\n"
            f" Model OK:   {'[green]Yes[/]' if model_loaded else '[red]No[/] (load a model first)'}\n"
        )

    def _endpoints_text(self) -> str:
        """Generate API endpoints documentation."""
        try:
            url = self.app.server.base_url
        except Exception:
            url = "http://localhost:1234"

        return (
            " [bold]API Endpoints (OpenAI-compatible)[/]\n"
            " ─────────────────────────────────────────\n\n"
            f" [bold cyan]GET[/]   {url}/v1/models\n"
            f"         List available models\n\n"
            f" [bold green]POST[/]  {url}/v1/chat/completions\n"
            f"         Chat completions (supports streaming)\n\n"
            f" [bold green]POST[/]  {url}/v1/completions\n"
            f"         Text completions (supports streaming)\n\n"
            f" [bold green]POST[/]  {url}/v1/embeddings\n"
            f"         Generate embeddings\n\n"
            f" [bold yellow]GET[/]   {url}/health\n"
            f"         Health check\n\n"
            " [dim]Usage with OpenAI Python client:[/]\n"
            f" [dim]  client = OpenAI(base_url=\"{url}/v1\", api_key=\"not-needed\")[/]\n"
            f" [dim]  response = client.chat.completions.create(model=\"local-model\", ...)[/]\n"
        )

    def on_mount(self):
        self._refresh_status()

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id == "btn-start-server":
            self._start_server()
        elif btn_id == "btn-stop-server":
            self._stop_server()
        elif btn_id == "btn-restart-server":
            self._stop_server()
            self._start_server()

    def _start_server(self):
        """Start the API server."""
        if self.app.server.is_running:
            self.app.notify("Server is already running", severity="warning")
            return

        if not self.app.engine.is_loaded:
            self.app.notify(
                "Load a model first before starting the server",
                severity="warning",
            )
            return

        # Update config from inputs
        try:
            host = self.query_one("#server-host-input", Input).value
            port = int(self.query_one("#server-port-input", Input).value)
            api_key = self.query_one("#server-apikey-input", Input).value.strip()

            self.app.config.server.host = host
            self.app.config.server.port = port
            self.app.config.server.api_key = api_key or None

            # Rebuild server with new config
            from llm_studio.server.api import LLMServer
            self.app.server = LLMServer(self.app.engine, self.app.config.server)
            self.app.server.start()

            self.app.notify(
                f"Server started at {self.app.server.base_url}",
            )
        except Exception as e:
            self.app.notify(f"Failed to start server: {e}", severity="error")

        self._refresh_status()

    def _stop_server(self):
        """Stop the API server."""
        if not self.app.server.is_running:
            self.app.notify("Server is not running", severity="warning")
            return

        self.app.server.stop()
        self.app.notify("Server stopped")
        self._refresh_status()

    def _refresh_status(self):
        """Update the displayed status."""
        try:
            self.query_one("#server-status-text", Static).update(
                self._status_text()
            )
            self.query_one("#endpoints-info", Static).update(
                self._endpoints_text()
            )
            # Update status bar
            sb = self.app.query_one("StatusBar")
            if self.app.server.is_running:
                sb.server_status = f"[green]Server: ON ({self.app.server.base_url})[/]"
            else:
                sb.server_status = "[red]Server: OFF[/]"
        except Exception:
            pass
