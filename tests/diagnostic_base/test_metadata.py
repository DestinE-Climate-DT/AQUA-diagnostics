"""Minimal tests for aqua.diagnostics.base.metadata."""

import xml.etree.ElementTree as ET

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from aqua.diagnostics.base.metadata import (
    add_figure_metadata,
    add_pdf_metadata,
    add_png_metadata,
    add_svg_metadata,
)

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_add_pdf_metadata_writes_and_normalizes_keys(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    add_pdf_metadata(str(pdf_path), {"Author": "AQUA", "/Title": "Diagnostics"})

    reader = PdfReader(str(pdf_path))
    assert reader.metadata["/Author"] == "AQUA"
    assert reader.metadata["/Title"] == "Diagnostics"


def test_add_png_metadata_writes_text_chunk(tmp_path):
    png_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), color="white").save(png_path, "PNG")

    add_png_metadata(str(png_path), {"author": "AQUA"})

    reopened = Image.open(png_path)
    assert reopened.info.get("author") == "AQUA"


def test_add_svg_metadata_inserts_desc(tmp_path):
    svg_path = tmp_path / "test.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>', encoding="utf-8")

    add_svg_metadata(str(svg_path), {"author": "AQUA", "version": "1"})

    root = ET.parse(svg_path).getroot()
    desc = root.find("{http://www.w3.org/2000/svg}desc")
    assert desc is not None
    assert "author: AQUA" in desc.text
    assert "version: 1" in desc.text


def test_metadata_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        add_pdf_metadata("/nonexistent/file.pdf", {})
    with pytest.raises(FileNotFoundError):
        add_png_metadata("/nonexistent/file.png", {})
    with pytest.raises(FileNotFoundError):
        add_svg_metadata("/nonexistent/file.svg", {})


def test_add_figure_metadata_dispatch(mocker):
    m_pdf = mocker.patch("aqua.diagnostics.base.metadata.add_pdf_metadata")
    m_png = mocker.patch("aqua.diagnostics.base.metadata.add_png_metadata")
    m_svg = mocker.patch("aqua.diagnostics.base.metadata.add_svg_metadata")

    add_figure_metadata("a.pdf", {"k": "v"}, "pdf")
    add_figure_metadata("a.png", {"k": "v"}, "png")
    add_figure_metadata("a.svg", {"k": "v"}, "svg")

    m_pdf.assert_called_once()
    m_png.assert_called_once()
    m_svg.assert_called_once()
