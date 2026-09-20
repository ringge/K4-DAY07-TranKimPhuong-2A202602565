"""Tests for the public-page HTML text extractor."""

from scripts.fetch_public_pages import extract_content


def test_extract_content_collapses_crlf_blank_lines():
    body = "<html><body>First\r\n\r\n\r\n\r\nSecond</body></html>"

    _, content = extract_content(body)

    assert content == "First\n\nSecond"


def test_extract_content_collapses_blank_lines_between_block_tags():
    body = """
        <html>
          <body>
            <div>First</div>
            <div>Second</div>
          </body>
        </html>
    """.replace("\n", "\r\n")

    _, content = extract_content(body)

    assert content == "First\n\nSecond"
    assert "\n\n\n" not in content
