from pathlib import Path

def test_frontend_files_exist():
    """Verify that the frontend UI files exist and are populated."""
    ui_dir = Path("ui")

    # Check that required files exist
    assert (ui_dir / "index.html").exists()
    assert (ui_dir / "style.css").exists()
    assert (ui_dir / "script.js").exists()

def test_frontend_html_structure():
    """Verify that index.html contains the basic required structure."""
    with open("ui/index.html", "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert "Nexus AI" in content
    assert "Ask Nexus AI..." in content
