"""Module containing functions necessary to add metadata to different output formats."""

import os
import xml.etree.ElementTree as ET

from PIL import Image, PngImagePlugin
from pypdf import PdfReader, PdfWriter

from aqua.core.logger import log_configure


def add_pdf_metadata(pdf_path: str, metadata: dict, loglevel: str = "WARNING"):
    """
    Open a PDF and write metadata.

    Args:
        pdf_path (str): Full path to the PDF file.
        metadata (dict): Metadata to write into the PDF. Keys may or may not
                         start with '/', they will be normalized.
        loglevel (str): the log level. Default is 'WARNING'.

    Raise:
        FileNotFoundError: if the file does not exist.
    """
    logger = log_configure(loglevel, "add_pdf_metadata")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found")

    pdf_reader = PdfReader(pdf_path)
    pdf_writer = PdfWriter()

    # Add existing pages to the new PDF
    for page in pdf_reader.pages:
        pdf_writer.add_page(page)

    # Normalize keys to start with '/' as required by PDF spec
    if metadata is None:
        metadata = {}
    metadata_normalized = {(k if isinstance(k, str) and k.startswith("/") else f"/{k}"): v for k, v in metadata.items()}
    pdf_writer.add_metadata(metadata_normalized)
    logger.debug(f"Metadata added to PDF: {pdf_path}")

    # Overwrite input PDF
    with open(pdf_path, "wb") as f:
        pdf_writer.write(f)


def add_png_metadata(png_path: str, metadata: dict, loglevel: str = "WARNING"):
    """
    Add metadata to a PNG image file.

    Args:
        png_path (str): The path to the PNG image file.
        metadata (dict): A dictionary of metadata to add to the PNG file.
                         Note: Metadata keys do not need a '/' prefix.
        loglevel (str): The log level. Default is 'WARNING'.
    """
    logger = log_configure(loglevel, "add_png_metadata")

    if not os.path.isfile(png_path):
        raise FileNotFoundError(f"File {png_path} not found")

    image = Image.open(png_path)

    # Create a dictionary for the PNG metadata
    png_info = PngImagePlugin.PngInfo()

    # Add the new metadata
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
        logger.debug(f"Adding metadata: {key} = {value}")

    # Save the file with the new metadata
    image.save(png_path, "PNG", pnginfo=png_info)
    logger.debug(f"Metadata added to PNG: {png_path}")


def add_svg_metadata(svg_path: str, metadata: dict, loglevel: str = "WARNING"):
    """
    Add metadata to an SVG image file.

    Args:
        svg_path (str): The path to the SVG image file.
        metadata (dict): A dictionary of metadata to add to the SVG file.
        loglevel (str): The log level. Default is 'WARNING'.
    """
    logger = log_configure(loglevel, "add_svg_metadata")

    if not os.path.isfile(svg_path):
        raise FileNotFoundError(f"File {svg_path} not found")

    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Check if desc already exists
        desc = root.find("{http://www.w3.org/2000/svg}desc")
        if desc is None:
            desc = ET.Element("{http://www.w3.org/2000/svg}desc")
            root.insert(0, desc)

        desc.text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        logger.debug(f"Metadata added to SVG: {svg_path}")
    except Exception as e:
        logger.warning(f"Failed to add metadata to SVG {svg_path}: {e}")


def add_figure_metadata(filepath: str, metadata: dict, file_format: str, loglevel: str = "WARNING"):
    """
    Add metadata to a figure file according to its format (PDF, PNG, or SVG).

    Args:
        filepath (str): Path to the figure file.
        metadata (dict): Metadata to embed.
        file_format (str): One of 'pdf', 'png', 'svg'.
        loglevel (str): Log level. Default is 'WARNING'.
    """
    if file_format == "pdf":
        add_pdf_metadata(filepath, metadata, loglevel=loglevel)
    elif file_format == "png":
        add_png_metadata(filepath, metadata, loglevel=loglevel)
    elif file_format == "svg":
        add_svg_metadata(filepath, metadata, loglevel=loglevel)
