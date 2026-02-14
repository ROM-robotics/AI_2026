"""Settings screen — configure inference parameters and app settings."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Button, Input, Switch, Select, Label
from textual.widget import Widget


class SettingsScreen(Widget):
    """Application settings and inference configuration screen."""

    DEFAULT_CSS = """
    SettingsScreen {
        width: 1fr;
        height: 1fr;
    }
    #settings-header {
        height: 3;
        background: $surface;
        border-bottom: tall $primary-background;
        padding: 0 2;
        content-align: left middle;
    }
    #settings-scroll {
        padding: 1 2;
    }
    .settings-section {
        margin: 1 0;
        padding: 1 2;
        background: $surface;
        border: round $primary-background;
    }
    .settings-row {
        height: 3;
        margin: 0 0 1 0;
    }
    .settings-label {
        width: 22;
        content-align: right middle;
        padding: 0 1;
    }
    .settings-input {
        width: 1fr;
    }
    .settings-hint {
        width: 1fr;
        color: $text-muted;
        padding: 0 1;
    }
    .settings-actions {
        height: 3;
        margin: 2 0;
    }
    .settings-actions Button {
        margin: 0 1 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            " ⚙️  [bold]Settings[/]",
            id="settings-header",
        )
        with VerticalScroll(id="settings-scroll"):
            # ── Inference settings ─────────────────────────────────
            with Vertical(classes="settings-section"):
                yield Static("[bold]Inference Parameters[/]\n")

                yield from self._setting_row(
                    "Context Length:", "setting-n-ctx",
                    str(self.app.config.inference.n_ctx),
                    "Max context window size (tokens)",
                )
                yield from self._setting_row(
                    "Max Tokens:", "setting-max-tokens",
                    str(self.app.config.inference.max_tokens),
                    "Max tokens to generate",
                )
                yield from self._setting_row(
                    "Temperature:", "setting-temperature",
                    str(self.app.config.inference.temperature),
                    "Randomness (0.0 = deterministic, 2.0 = very random)",
                )
                yield from self._setting_row(
                    "Top P:", "setting-top-p",
                    str(self.app.config.inference.top_p),
                    "Nucleus sampling threshold",
                )
                yield from self._setting_row(
                    "Top K:", "setting-top-k",
                    str(self.app.config.inference.top_k),
                    "Top-K sampling",
                )
                yield from self._setting_row(
                    "Repeat Penalty:", "setting-repeat-penalty",
                    str(self.app.config.inference.repeat_penalty),
                    "Penalize repeated tokens",
                )
                yield from self._setting_row(
                    "Threads:", "setting-n-threads",
                    str(self.app.config.inference.n_threads),
                    "CPU threads for inference",
                )
                yield from self._setting_row(
                    "GPU Layers:", "setting-n-gpu-layers",
                    str(self.app.config.inference.n_gpu_layers),
                    "Layers to offload to GPU (0 = CPU only)",
                )
                yield from self._setting_row(
                    "Batch Size:", "setting-n-batch",
                    str(self.app.config.inference.n_batch),
                    "Prompt processing batch size",
                )
                yield from self._setting_row(
                    "Seed:", "setting-seed",
                    str(self.app.config.inference.seed),
                    "Random seed (-1 = random)",
                )

            # ── System prompt ──────────────────────────────────────
            with Vertical(classes="settings-section"):
                yield Static("[bold]System Prompt[/]\n")
                yield Input(
                    value=self.app.config.system_prompt,
                    placeholder="Enter system prompt...",
                    id="setting-system-prompt",
                )

            # ── Paths ──────────────────────────────────────────────
            with Vertical(classes="settings-section"):
                yield Static("[bold]Storage[/]\n")
                yield from self._setting_row(
                    "Models Dir:", "setting-models-dir",
                    self.app.config.models_dir,
                    "Directory for downloaded models",
                )

            # ── Actions ────────────────────────────────────────────
            with Horizontal(classes="settings-actions"):
                yield Button(
                    "💾  Save Settings",
                    id="btn-save-settings",
                    variant="success",
                )
                yield Button(
                    "↩  Reset Defaults",
                    id="btn-reset-settings",
                    variant="warning",
                )

    def _setting_row(self, label: str, input_id: str, value: str, hint: str):
        """Generate a settings row with label, input, and hint."""
        with Horizontal(classes="settings-row"):
            yield Static(label, classes="settings-label")
            yield Input(value=value, id=input_id, classes="settings-input")
        yield Static(f"    [dim]{hint}[/]", classes="settings-hint")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-save-settings":
            self._save_settings()
        elif event.button.id == "btn-reset-settings":
            self._reset_settings()

    def _save_settings(self):
        """Save all settings to config."""
        config = self.app.config

        try:
            config.inference.n_ctx = int(
                self.query_one("#setting-n-ctx", Input).value
            )
            config.inference.max_tokens = int(
                self.query_one("#setting-max-tokens", Input).value
            )
            config.inference.temperature = float(
                self.query_one("#setting-temperature", Input).value
            )
            config.inference.top_p = float(
                self.query_one("#setting-top-p", Input).value
            )
            config.inference.top_k = int(
                self.query_one("#setting-top-k", Input).value
            )
            config.inference.repeat_penalty = float(
                self.query_one("#setting-repeat-penalty", Input).value
            )
            config.inference.n_threads = int(
                self.query_one("#setting-n-threads", Input).value
            )
            config.inference.n_gpu_layers = int(
                self.query_one("#setting-n-gpu-layers", Input).value
            )
            config.inference.n_batch = int(
                self.query_one("#setting-n-batch", Input).value
            )
            config.inference.seed = int(
                self.query_one("#setting-seed", Input).value
            )
            config.system_prompt = self.query_one(
                "#setting-system-prompt", Input
            ).value
            config.models_dir = self.query_one(
                "#setting-models-dir", Input
            ).value

            # Update engine config
            self.app.engine.config = config.inference

            config.save()
            self.app.notify("Settings saved!", severity="information")

        except ValueError as e:
            self.app.notify(f"Invalid value: {e}", severity="error")

    def _reset_settings(self):
        """Reset all settings to defaults."""
        from llm_studio.config import AppConfig
        default = AppConfig()
        self.app.config = default

        # Update input fields
        field_map = {
            "#setting-n-ctx": str(default.inference.n_ctx),
            "#setting-max-tokens": str(default.inference.max_tokens),
            "#setting-temperature": str(default.inference.temperature),
            "#setting-top-p": str(default.inference.top_p),
            "#setting-top-k": str(default.inference.top_k),
            "#setting-repeat-penalty": str(default.inference.repeat_penalty),
            "#setting-n-threads": str(default.inference.n_threads),
            "#setting-n-gpu-layers": str(default.inference.n_gpu_layers),
            "#setting-n-batch": str(default.inference.n_batch),
            "#setting-seed": str(default.inference.seed),
            "#setting-system-prompt": default.system_prompt,
            "#setting-models-dir": default.models_dir,
        }
        for sel, val in field_map.items():
            try:
                self.query_one(sel, Input).value = val
            except Exception:
                pass

        self.app.notify("Settings reset to defaults", severity="information")
