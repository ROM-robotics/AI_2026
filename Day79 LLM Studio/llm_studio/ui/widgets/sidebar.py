"""Sidebar navigation widget."""

from textual.app import ComposeResult
from textual.widgets import Static, Button, Label
from textual.containers import Vertical
from textual.widget import Widget
from textual.reactive import reactive


class SidebarItem(Button):
    """A navigation item in the sidebar."""

    DEFAULT_CSS = """
    SidebarItem {
        width: 100%;
        height: 3;
        background: transparent;
        border: none;
        content-align: left middle;
        padding: 0 1;
        margin: 0;
        text-style: none;
    }
    SidebarItem:hover {
        background: $surface-lighten-1;
    }
    SidebarItem.-active {
        background: $primary;
        color: $text;
        text-style: bold;
    }
    """

    def __init__(self, label: str, icon: str, screen_name: str, **kwargs):
        super().__init__(f" {icon}  {label}", **kwargs)
        self.screen_name = screen_name


class Sidebar(Widget):
    """Left sidebar navigation for LLM Studio."""

    DEFAULT_CSS = """
    Sidebar {
        width: 28;
        dock: left;
        background: $surface;
        border-right: tall $primary-background;
        padding: 1 0;
    }
    """

    active_screen: reactive[str] = reactive("home")

    NAVIGATION = [
        ("Home", "🏠", "home"),
        ("Chat", "💬", "chat"),
        ("Models", "📦", "models"),
        ("Server", "🌐", "server"),
        ("Settings", "⚙️ ", "settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "┌─────────────────────────┐\n"
            "│    ◆ LLM Studio  v1.0   │\n"
            "└─────────────────────────┘",
            classes="sidebar-logo",
        )
        yield Static("─" * 26, classes="sidebar-divider")
        with Vertical(id="nav-items"):
            for label, icon, screen in self.NAVIGATION:
                item = SidebarItem(label, icon, screen, id=f"nav-{screen}")
                if screen == self.active_screen:
                    item.add_class("-active")
                yield item
        yield Static("─" * 26, classes="sidebar-divider")
        yield Static(
            "  [dim]Ctrl+Q  Quit[/]\n"
            "  [dim]Ctrl+T  Theme[/]\n"
            "  [dim]Tab     Navigate[/]",
            classes="sidebar-help",
        )

    def watch_active_screen(self, value: str):
        """Update active highlight when screen changes."""
        for item in self.query(SidebarItem):
            if item.screen_name == value:
                item.add_class("-active")
            else:
                item.remove_class("-active")
