"""Models screen — browse, download, load, and manage GGUF models."""

import threading
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Static, Button, Input, Label, DataTable, ProgressBar,
    TabbedContent, TabPane,
)
from textual.widget import Widget


class ModelsScreen(Widget):
    """Model management screen."""

    DEFAULT_CSS = """
    ModelsScreen {
        width: 1fr;
        height: 1fr;
    }
    #models-header {
        height: 3;
        background: $surface;
        border-bottom: tall $primary-background;
        padding: 0 2;
        content-align: left middle;
    }
    .model-section {
        padding: 1 2;
    }
    #local-models-table {
        height: 1fr;
        margin: 1 0;
    }
    #hf-search-bar {
        height: 3;
        padding: 0 0;
        margin: 1 0;
    }
    #hf-search-input {
        width: 1fr;
    }
    #btn-search {
        width: 14;
        margin: 0 0 0 1;
    }
    #hf-results-table {
        height: 1fr;
        margin: 1 0;
    }
    #download-section {
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $primary-background;
        margin: 1 0;
    }
    .model-actions {
        height: 3;
        margin: 1 0;
    }
    .model-actions Button {
        margin: 0 1 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            " 📦 [bold]Model Manager[/]",
            id="models-header",
        )
        with TabbedContent():
            with TabPane("Local Models", id="tab-local"):
                yield from self._local_tab()
            with TabPane("Download from HuggingFace", id="tab-download"):
                yield from self._download_tab()

    def _local_tab(self):
        """Local models management tab."""
        with Vertical(classes="model-section"):
            with Horizontal(classes="model-actions"):
                yield Button("🔄  Refresh", id="btn-refresh-models", variant="default")
                yield Button("▶  Load Selected", id="btn-load-model", variant="primary")
                yield Button("⏹  Unload Model", id="btn-unload-model", variant="warning")
                yield Button("🗑  Delete Selected", id="btn-delete-model", variant="error")

            table = DataTable(id="local-models-table")
            table.cursor_type = "row"
            yield table

            yield Static("", id="local-models-info")

    def _download_tab(self):
        """HuggingFace download tab."""
        with Vertical(classes="model-section"):
            yield Static(
                " Search for GGUF models on HuggingFace\n"
                " [dim]Example: TheBloke/Mistral-7B-Instruct-v0.2-GGUF[/]",
            )
            with Horizontal(id="hf-search-bar"):
                yield Input(
                    placeholder="Enter repo ID or search query...",
                    id="hf-search-input",
                )
                yield Button("🔍  Search", id="btn-search", variant="primary")

            table = DataTable(id="hf-results-table")
            table.cursor_type = "row"
            yield table

            with Horizontal(classes="model-actions"):
                yield Button(
                    "⬇  Download Selected",
                    id="btn-download-model",
                    variant="success",
                )

            yield ProgressBar(id="download-progress", total=100, show_eta=True)
            yield Static("", id="download-status")

    def on_mount(self):
        """Initialize tables."""
        self._setup_local_table()
        self._setup_hf_table()
        self._refresh_local_models()

    def _setup_local_table(self):
        table = self.query_one("#local-models-table", DataTable)
        table.add_columns("Filename", "Size", "Quantization", "Status")

    def _setup_hf_table(self):
        table = self.query_one("#hf-results-table", DataTable)
        table.add_columns("Filename", "Size", "Repo")

    def _refresh_local_models(self):
        """Refresh the local models list."""
        table = self.query_one("#local-models-table", DataTable)
        table.clear()

        manager = self.app.model_manager
        models = manager.list_local_models()

        current = None
        if self.app.engine.is_loaded:
            current = Path(self.app.engine.model_path).name

        for m in models:
            status = "● Loaded" if m.filename == current else ""
            table.add_row(
                m.filename,
                m.size_display,
                m.quantization,
                status,
                key=m.filename,
            )

        info = manager.get_storage_usage()
        size_gb = info["total_size"] / (1024 ** 3)
        free_gb = info["disk_free"] / (1024 ** 3)
        self.query_one("#local-models-info", Static).update(
            f" [dim]{info['model_count']} models  •  "
            f"{size_gb:.2f} GB used  •  "
            f"{free_gb:.1f} GB free[/]"
        )

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id == "btn-refresh-models":
            self._refresh_local_models()
            self.app.notify("Models refreshed")

        elif btn_id == "btn-load-model":
            self._load_selected_model()

        elif btn_id == "btn-unload-model":
            self._unload_model()

        elif btn_id == "btn-delete-model":
            self._delete_selected_model()

        elif btn_id == "btn-search":
            self._search_hf()

        elif btn_id == "btn-download-model":
            self._download_selected()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "hf-search-input":
            self._search_hf()

    def _load_selected_model(self):
        """Load the selected model from the table."""
        table = self.query_one("#local-models-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            self.app.notify("No model selected", severity="warning")
            return

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        filename = str(row_key.value)
        model_path = self.app.model_manager.get_model_path(filename)

        if not model_path:
            self.app.notify("Model file not found", severity="error")
            return

        self.app.notify(f"Loading {filename}...")
        self.query_one("#local-models-info", Static).update(
            f" [bold yellow]Loading {filename}...[/]"
        )

        def _load():
            try:
                self.app.engine.load_model(model_path)
                self.app.call_from_thread(
                    self.app.notify, f"Model loaded: {filename}"
                )
                self.app.call_from_thread(self._refresh_local_models)
                self.app.call_from_thread(self._update_status_bar)
            except Exception as e:
                self.app.call_from_thread(
                    self.app.notify,
                    f"Failed to load: {e}",
                    severity="error",
                )

        threading.Thread(target=_load, daemon=True).start()

    def _unload_model(self):
        """Unload the current model."""
        if not self.app.engine.is_loaded:
            self.app.notify("No model loaded", severity="warning")
            return
        self.app.engine.unload_model()
        self._refresh_local_models()
        self._update_status_bar()
        self.app.notify("Model unloaded")

    def _delete_selected_model(self):
        """Delete the selected model file."""
        table = self.query_one("#local-models-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            self.app.notify("No model selected", severity="warning")
            return

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        filename = str(row_key.value)

        # Unload first if this model is loaded
        if self.app.engine.is_loaded:
            current = Path(self.app.engine.model_path).name
            if current == filename:
                self.app.engine.unload_model()

        if self.app.model_manager.delete_model(filename):
            self._refresh_local_models()
            self._update_status_bar()
            self.app.notify(f"Deleted: {filename}")
        else:
            self.app.notify("Failed to delete model", severity="error")

    def _search_hf(self):
        """Search HuggingFace for GGUF files."""
        query = self.query_one("#hf-search-input", Input).value.strip()
        if not query:
            self.app.notify("Enter a search query", severity="warning")
            return

        self.app.notify(f"Searching: {query}...")

        def _do_search():
            manager = self.app.model_manager
            table = self.query_one("#hf-results-table", DataTable)

            # Check if it looks like a repo ID
            if "/" in query:
                files = manager.list_gguf_files(query)
                self.app.call_from_thread(table.clear)
                for f in files:
                    self.app.call_from_thread(
                        table.add_row,
                        f.filename, f.size_display, f.repo_id,
                        key=f"{f.repo_id}|||{f.filename}",
                    )
                self.app.call_from_thread(
                    self.app.notify,
                    f"Found {len(files)} GGUF files",
                )
            else:
                results = manager.search_hf_models(query)
                self.app.call_from_thread(table.clear)
                for r in results:
                    self.app.call_from_thread(
                        table.add_row,
                        r["repo_id"],
                        f"↓{r['downloads']}",
                        f"♥{r['likes']}",
                        key=r["repo_id"],
                    )
                self.app.call_from_thread(
                    self.app.notify,
                    f"Found {len(results)} repos (enter repo ID for GGUF files)",
                )

        threading.Thread(target=_do_search, daemon=True).start()

    def _download_selected(self):
        """Download the selected GGUF file."""
        table = self.query_one("#hf-results-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            self.app.notify("No file selected", severity="warning")
            return

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        key_str = str(row_key.value)

        if "|||" not in key_str:
            self.app.notify(
                "Select a specific GGUF file (search with repo ID first)",
                severity="warning",
            )
            return

        repo_id, filename = key_str.split("|||", 1)
        progress_bar = self.query_one("#download-progress", ProgressBar)
        status_label = self.query_one("#download-status", Static)

        status_label.update(f" [bold yellow]Downloading {filename}...[/]")
        self.app.notify(f"Downloading {filename}...")

        def _download():
            def prog(pct):
                self.app.call_from_thread(progress_bar.update, progress=pct)

            manager = self.app.model_manager
            path = manager.download_model(repo_id, filename, progress_callback=prog)

            if path:
                self.app.call_from_thread(
                    status_label.update,
                    f" [bold green]✓ Downloaded: {filename}[/]"
                )
                self.app.call_from_thread(
                    self.app.notify,
                    f"Download complete: {filename}",
                )
                self.app.call_from_thread(self._refresh_local_models)
            else:
                self.app.call_from_thread(
                    status_label.update,
                    f" [bold red]✗ Download failed[/]"
                )
                self.app.call_from_thread(
                    self.app.notify,
                    "Download failed",
                    severity="error",
                )

        threading.Thread(target=_download, daemon=True).start()

    def _update_status_bar(self):
        """Update the app status bar with current model info."""
        try:
            sb = self.app.query_one("StatusBar")
            if self.app.engine.is_loaded:
                name = Path(self.app.engine.model_path).stem
                sb.model_name = name
            else:
                sb.model_name = "No model loaded"
        except Exception:
            pass
