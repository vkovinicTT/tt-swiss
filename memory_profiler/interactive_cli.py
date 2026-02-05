# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive CLI tool for TT Memory Profiler.

Provides a user-friendly interface for processing memory logs
and generating HTML reports.

Supports both interactive mode and command-line mode:
  ttmem                           # Interactive mode
  ttmem --llm --logfile log.log   # LLM report to stdout
  ttmem --llm --logfile log.log -o report.md  # LLM report to file
"""

import argparse
import http.server
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

# Handle both package import and direct execution
try:
    from .parser import parse_log_file, validate_outputs
    from .text_formatter import LLMTextFormatter
    from .run_profiled import sanitize_report_name, get_reports_dir
except ImportError:
    from parser import parse_log_file, validate_outputs
    from text_formatter import LLMTextFormatter
    from run_profiled import sanitize_report_name, get_reports_dir


def _import_interactive_deps():
    """Import dependencies needed for interactive mode. Returns True if successful."""
    global Console, Panel, Spinner, Live, inquirer, Choice, InquirerPyStyle
    global MemoryVisualizer, CUSTOM_STYLE, console

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.spinner import Spinner
        from rich.live import Live
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
        from InquirerPy.utils import InquirerPyStyle
    except ImportError:
        return False

    try:
        from .visualizer import MemoryVisualizer
    except ImportError:
        from visualizer import MemoryVisualizer

    # Custom style for InquirerPy prompts (cyan/blue theme matching the logo)
    CUSTOM_STYLE = InquirerPyStyle({
        "questionmark": "fg:cyan bold",
        "question": "bold",
        "answer": "fg:cyan bold",
        "pointer": "fg:cyan bold",
        "highlighted": "fg:cyan bold",
        "selected": "fg:cyan",
        "instruction": "fg:gray",
    })

    console = Console()
    return True

# ASCII Logo using only - and | characters
LOGO = r"""
 |----|  |----|  |\    /|  |-----  |\    /|
   ||      ||    | \  / |  |       | \  / |
   ||      ||    |  \/  |  |----   |  \/  |
   ||      ||    |      |  |       |      |
   ||      ||    |      |  |-----  |      |
"""

# console is initialized by _import_interactive_deps() for interactive mode
console = None


def display_intro() -> None:
    """Display the ASCII logo and welcome message."""
    console.print(
        Panel(
            f"[cyan]{LOGO}[/cyan]\n[bold]Welcome to ttmem![/bold] Memory profiling made easy.",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def display_instructions() -> None:
    """Display prerequisites for generating memory logs."""
    instructions = """
[bold yellow]Prerequisites for Memory Logging:[/bold yellow]

[bold]1. Enable memory logging in TT-XLA:[/bold]
   Add this to your TT-XLA code:
   [cyan]tt::runtime::setMemoryLogLevel(tt::runtime::MemoryLogLevel::Operation)[/cyan]

[bold]2. Build TT-XLA with debug flags:[/bold]
   [cyan]-DCMAKE_BUILD_TYPE=Debug -DTT_RUNTIME_DEBUG=ON[/cyan]

[bold]3. Set the environment variable:[/bold]
   [cyan]export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG[/cyan]

[bold]4. Run your model script:[/bold]
   [cyan]tt-memory-profiler --log path/to/your_model.py[/cyan]

   This will generate a log file at:
   [cyan]./logs/<script_name>_YYYYMMDD_HHMMSS/<script_name>_profile.log[/cyan]
"""
    console.print(
        Panel(
            instructions,
            title="[bold]Getting Started[/bold]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def ask_has_log_file() -> Optional[str]:
    """Ask the user if they already have a log file.

    Returns:
        'yes' - User has a log file ready
        'do_it_for_me' - User wants automated log generation
        'instructions' - User wants step-by-step instructions
        'browse' - User wants to browse existing reports
        None - User cancelled (Q pressed)
    """
    return inquirer.select(
        message="Do you have a log file with memory logs ready?",
        choices=[
            Choice(value="yes", name="Yes, I have a log file with memory logs ready."),
            Choice(value="do_it_for_me", name="No, do it for me."),
            Choice(value="instructions", name="No, give me step-by-step so I can do it myself."),
            Choice(value="browse", name="Browse existing reports"),
        ],
        instruction="(Arrow keys to navigate, Enter to select, Q to quit)",
        style=CUSTOM_STYLE,
        mandatory=False,
        keybindings={"skip": [{"key": "q"}, {"key": "Q"}]},
    ).execute()


def wait_for_ready() -> None:
    """Wait for the user to press Enter when ready."""
    console.print()
    input("Press Enter when you have your log file ready...")


def do_it_for_me_placeholder() -> None:
    """Placeholder for automated log generation feature."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]Feature Coming Soon![/bold yellow]\n\n"
            "Automated log generation is not yet implemented.\n"
            "This feature will automatically run your model script\n"
            "with the correct environment settings and generate logs.",
            title="[bold]Do It For Me[/bold]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()
    input("Press Enter to return to the main menu...")


def ask_log_file_path() -> Optional[str]:
    """Prompt the user for the log file path with autocomplete."""
    console.print()
    console.print("[dim]Q to go back[/dim]")
    return inquirer.filepath(
        message="Enter the path to your log file:",
        validate=lambda p: validate_log_path(p) is None,
        invalid_message="Invalid path. File must exist and be readable.",
        style=CUSTOM_STYLE,
        mandatory=False,
        keybindings={"skip": [{"key": "q"}, {"key": "Q"}]},
    ).execute()


def validate_log_path(path: str) -> Optional[str]:
    """
    Validate the log file path.

    Returns None if valid, or an error message string if invalid.
    """
    if not path:
        return "Path cannot be empty"

    file_path = Path(path)

    if not file_path.exists():
        return f"File not found: {path}"

    if not file_path.is_file():
        return f"Not a file: {path}"

    if not os.access(file_path, os.R_OK):
        return f"File is not readable: {path}"

    # Check if file has content
    if file_path.stat().st_size == 0:
        return f"File is empty: {path}"

    return None


def process_log_file(log_path: str) -> Optional[Path]:
    """
    Process the log file and generate the HTML report.

    Reports are saved to ~/.ttmem/reports/<report_name>/ with sanitized names.

    Returns the path to the generated report, or None on failure.
    """
    log_file = Path(log_path)

    # Sanitize log file name and save to ~/.ttmem/reports/<report_name>/
    report_name = sanitize_report_name(log_file.stem)
    run_dir = get_reports_dir() / report_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths using sanitized report name
    mem_output = run_dir / f"{report_name}_memory.json"
    ops_output = run_dir / f"{report_name}_operations.json"
    registry_output = run_dir / f"{report_name}_inputs_registry.json"
    ir_output = run_dir / f"{report_name}_ir.json"

    console.print()
    console.print(f"[dim]Output directory: {run_dir}[/dim]")

    # Step 1: Parse log file
    with console.status("[bold cyan]Parsing log file...", spinner="dots") as status:
        try:
            parse_log_file(
                str(log_file),
                str(mem_output),
                str(ops_output),
                str(registry_output),
                str(ir_output),
            )
        except Exception as e:
            console.print(f"[red]Error parsing log file: {e}[/red]")
            return None

    # Step 2: Validate outputs
    with console.status("[bold cyan]Validating outputs...", spinner="dots") as status:
        if not validate_outputs(str(mem_output), str(ops_output)):
            console.print("[red]Output validation failed[/red]")
            return None

    # Step 3: Generate visualization
    with console.status("[bold cyan]Generating HTML report...", spinner="dots") as status:
        try:
            visualizer = MemoryVisualizer(run_dir, script_name=report_name)
            report_path = visualizer.generate_report()
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")
            return None

    return report_path


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def start_http_server(directory: Path, port: int) -> Tuple[socketserver.TCPServer, threading.Thread]:
    """Start an HTTP server in the background serving the given directory.

    Binds to 0.0.0.0 so VS Code Remote SSH can auto-forward the port.
    """
    handler = http.server.SimpleHTTPRequestHandler

    class QuietHandler(handler):
        """HTTP handler that suppresses log messages."""
        def log_message(self, format, *args):
            pass  # Suppress logging

    os.chdir(directory)
    # Bind to 0.0.0.0 to allow connections from any interface
    # This enables VS Code's automatic port forwarding for remote development
    server = socketserver.TCPServer(("0.0.0.0", port), QuietHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server, thread


def browse_existing_reports() -> None:
    """Browse and serve existing reports from ~/.ttmem/reports/."""
    reports_dir = get_reports_dir()

    if not reports_dir.exists():
        console.print(
            Panel(
                "[yellow]No reports directory found.[/yellow]\n\n"
                "Process a log file first to generate reports.",
                title="[bold]No Reports[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()
        input("Press Enter to return to the main menu...")
        return

    # Find all report directories that contain HTML files
    report_dirs = []
    for item in sorted(reports_dir.iterdir()):
        if item.is_dir():
            html_files = list(item.glob("*.html"))
            if html_files:
                report_dirs.append((item.name, html_files[0]))

    if not report_dirs:
        console.print(
            Panel(
                "[yellow]No reports found in ~/.ttmem/reports/[/yellow]\n\n"
                "Process a log file first to generate reports.",
                title="[bold]No Reports[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()
        input("Press Enter to return to the main menu...")
        return

    # Create choices for the selection
    choices = [Choice(value=html_path, name=name) for name, html_path in report_dirs]

    console.print()
    console.print("[dim]Q to go back[/dim]")
    selected = inquirer.select(
        message="Select a report to view:",
        choices=choices,
        instruction="(Arrow keys to navigate, Enter to select, Q to go back)",
        style=CUSTOM_STYLE,
        mandatory=False,
        keybindings={"skip": [{"key": "q"}, {"key": "Q"}]},
    ).execute()

    if selected is None:
        return

    # Serve the selected report
    report_path = Path(selected)
    server = display_success(report_path)

    # If server is running, wait for Ctrl+C
    if server:
        try:
            console.print("\n[dim]Press Ctrl+C to stop the server and return to menu.[/dim]")
            while True:
                pass
        except KeyboardInterrupt:
            server.shutdown()
            console.print("\n[yellow]Server stopped.[/yellow]\n")


def display_success(report_path: Path) -> Optional[socketserver.TCPServer]:
    """
    Display success message with HTTP URL for the report.

    Always starts an HTTP server and returns the server instance.
    """
    server = None
    try:
        port = find_available_port()
        server, _ = start_http_server(report_path.parent, port)
        # Use localhost since VS Code Remote SSH will auto-forward the port
        http_url = f"http://localhost:{port}/{report_path.name}"

        success_message = f"""
[bold green]Report generated successfully![/bold green]

[bold]Open in browser:[/bold]
[link={http_url}]{http_url}[/link]

[dim]Server running on port {port}. Press Ctrl+C to stop.[/dim]
"""
    except Exception as e:
        success_message = f"""
[bold green]Report generated successfully![/bold green]

[yellow]Could not start HTTP server: {e}[/yellow]
[dim]You can manually serve the file with:[/dim]
[cyan]cd {report_path.parent} && python -m http.server[/cyan]
"""

    console.print(
        Panel(
            success_message,
            title="[bold green]Success[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    return server


def generate_llm_report(log_path: str, output_file: Optional[Path] = None) -> int:
    """
    Generate LLM-friendly text report from a log file.

    Args:
        log_path: Path to the log file to analyze
        output_file: Optional output file path. If None, prints to stdout.

    Returns:
        0 on success, 1 on failure
    """
    log_file = Path(log_path)

    # Validate log file
    error = validate_log_path(log_path)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Sanitize log file name and save to ~/.ttmem/reports/<report_name>/
    report_name = sanitize_report_name(log_file.stem)
    run_dir = get_reports_dir() / report_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    mem_output = run_dir / f"{report_name}_memory.json"
    ops_output = run_dir / f"{report_name}_operations.json"
    registry_output = run_dir / f"{report_name}_inputs_registry.json"
    ir_output = run_dir / f"{report_name}_ir.json"

    # Parse log file (suppress output when generating to stdout)
    try:
        if output_file:
            print(f"Parsing log file: {log_file}", file=sys.stderr)
            print(f"Output directory: {run_dir}", file=sys.stderr)

        parse_log_file(
            str(log_file),
            str(mem_output),
            str(ops_output),
            str(registry_output),
            str(ir_output),
        )
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        return 1

    # Validate outputs
    if not validate_outputs(str(mem_output), str(ops_output)):
        print("Error: Output validation failed", file=sys.stderr)
        return 1

    # Generate LLM report
    try:
        formatter = LLMTextFormatter(run_dir, script_name=report_name)
        report = formatter.generate_report(output_file=output_file)

        if output_file:
            print(f"LLM report written to: {output_file}", file=sys.stderr)
        else:
            print(report)

        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error generating LLM report: {e}", file=sys.stderr)
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TT Memory Profiler - Interactive CLI for memory profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  ttmem

  # LLM-friendly text report (output to stdout)
  ttmem --llm --logfile path/to/log.log

  # LLM-friendly text report (output to file)
  ttmem --llm --logfile path/to/log.log -o report.md
        """,
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Output LLM-friendly text report instead of HTML",
    )
    parser.add_argument(
        "--logfile",
        metavar="LOG_FILE",
        help="Log file to analyze (use with --llm)",
    )
    parser.add_argument(
        "-o", "--output-file",
        metavar="FILE",
        help="Write LLM report to file instead of stdout (use with --llm)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the interactive CLI."""
    args = parse_args()

    # Handle --llm mode (no rich/InquirerPy needed)
    if args.llm:
        if not args.logfile:
            print("Error: --llm requires --logfile", file=sys.stderr)
            return 1
        output_file = Path(args.output_file) if args.output_file else None
        return generate_llm_report(args.logfile, output_file)

    # Interactive mode - requires rich and InquirerPy
    if not _import_interactive_deps():
        print("Error: Required packages not found for interactive mode. Please install with:")
        print("  pip install rich InquirerPy")
        print("\nAlternatively, use --llm mode which doesn't require these packages:")
        print("  ttmem --llm --logfile path/to/log.log")
        return 1

    try:
        # Display intro
        display_intro()

        # Main menu loop
        while True:
            # Ask if user has a log file
            choice = ask_has_log_file()

            if choice is None:
                # User pressed Esc - quit
                console.print("\n[yellow]Goodbye![/yellow]")
                return 0

            if choice == "do_it_for_me":
                # Show placeholder and return to main menu
                do_it_for_me_placeholder()
                continue

            if choice == "browse":
                # Browse existing reports
                browse_existing_reports()
                continue

            if choice == "instructions":
                # Show instructions and wait for user
                display_instructions()
                wait_for_ready()

            # choice is 'yes' or 'instructions' (after waiting)
            # Get log file path
            log_path = ask_log_file_path()

            if log_path is None:
                # User pressed Esc - go back to main menu
                console.print("\n[dim]Returning to main menu...[/dim]\n")
                continue

            # Valid path provided, exit the loop
            break

        # Validate path (InquirerPy already validated, but double-check)
        error = validate_log_path(log_path)
        if error:
            console.print(f"[red]{error}[/red]")
            return 1

        # Process the log file
        report_path = process_log_file(log_path)

        if report_path is None:
            return 1

        # Display success and start server
        server = display_success(report_path)

        # If server is running, wait for Ctrl+C
        if server:
            try:
                console.print("\n[dim]Press Ctrl+C to stop the server and exit.[/dim]")
                while True:
                    pass
            except KeyboardInterrupt:
                server.shutdown()
                console.print("\n[yellow]Server stopped.[/yellow]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
