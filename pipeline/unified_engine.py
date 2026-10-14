"""Unified Document Processing Engine for BE-OCR Core.

This module provides a unified pipeline used across both Flask REST API
and Streamlit Dashboard. It safely converts input images/PDFs into PageLayout
representations and builds final DOCX documents.
"""

from __future__ import annotations

import io
import os
import uuid
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Union, Callable

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from pipeline.document_reconstructor import DocumentReconstructor, PageLayout
from pipeline.docx_builder import DocxBuilder
from pipeline.language_detector import FastLanguageDetector

logger = logging.getLogger(__name__)

# Security & Processing Constraints
DEFAULT_MAX_PDF_PAGES = 50
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000  # e.g., max ~6000x6000px to prevent decompression bombs


class DocumentProcessingError(Exception):
    """Custom exception for unified processing failures."""
    pass


class SecurityValidationError(DocumentProcessingError):
    """Raised when file dimensions, pages, or content fail security checks."""
    pass


class UnifiedDocumentEngine:
    def __init__(
        self,
        lang: str = "vi",
        artifacts_dir: str = "artifacts/reconstruct",
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ):
        self.lang = lang
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.max_pdf_pages = max_pdf_pages
        self.max_image_pixels = max_image_pixels

        # Thread safety lock for underlying inference calls (e.g. PaddleOCR)
        self._inference_lock = threading.Lock()

        # Lazy-loaded / persistent pipeline components
        self._reconstructor: Optional[DocumentReconstructor] = None
        self._docx_builder: Optional[DocxBuilder] = None
        self._lang_detector: Optional[FastLanguageDetector] = None

    @property
    def reconstructor(self) -> DocumentReconstructor:
        if self._reconstructor is None:
            self._reconstructor = DocumentReconstructor(
                output_artifacts_dir=str(self.artifacts_dir),
                lang=self.lang,
            )
        return self._reconstructor

    @property
    def docx_builder(self) -> DocxBuilder:
        if self._docx_builder is None:
            self._docx_builder = DocxBuilder()
        return self._docx_builder

    @property
    def lang_detector(self) -> FastLanguageDetector:
        if self._lang_detector is None:
            self._lang_detector = FastLanguageDetector()
        return self._lang_detector

    def iter_pdf_pages(self, pdf_bytes: bytes, dpi: int = 150):
        """Iteratively yield BGR numpy image arrays from PDF bytes.
        
        Processes pages one-by-one to prevent high memory consumption (OOM)
        on multi-page documents.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise SecurityValidationError(f"Invalid or corrupted PDF file: {e}")

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise SecurityValidationError("PDF file contains no pages.")
            if total_pages > self.max_pdf_pages:
                raise SecurityValidationError(
                    f"PDF exceeds page limit ({total_pages} > {self.max_pdf_pages} pages)."
                )

            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                total_px = pix.width * pix.height
                if total_px > self.max_image_pixels:
                    raise SecurityValidationError(
                        f"Page {page_idx + 1} exceeds max allowed pixel size ({total_px} > {self.max_image_pixels})."
                    )
                # PyMuPDF samples are RGB; convert directly to BGR for OpenCV / DocumentReconstructor
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                # Free pixmap explicitly before yielding
                del pix
                yield page_idx, img_bgr
        finally:
            doc.close()

    def rasterize_pdf_bytes(self, pdf_bytes: bytes, dpi: int = 150) -> List[np.ndarray]:
        """Convert PDF bytes to a list of BGR numpy image arrays (convenience wrapper)."""
        return [img for _, img in self.iter_pdf_pages(pdf_bytes, dpi=dpi)]

    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes to BGR numpy array with decompression bomb protection."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                total_px = pil_img.width * pil_img.height
                if total_px > self.max_image_pixels:
                    raise SecurityValidationError(
                        f"Image exceeds max allowed dimensions ({total_px} > {self.max_image_pixels} pixels)."
                    )
                # Ensure 3-channel RGB
                pil_rgb = pil_img.convert("RGB")
                img_rgb = np.array(pil_rgb)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr
        except SecurityValidationError:
            raise
        except Exception as e:
            raise SecurityValidationError(f"Failed to decode image: {e}")

    def process_to_layouts(
        self,
        file_bytes: bytes,
        file_kind: str,
        dpi: int = 150,
        page_transform_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PageLayout]:
        """Process file bytes to a list of PageLayout objects.
        
        Shared across API and Streamlit Dashboard. Allows optional per-page image
        transformation (e.g. red seal filter) and progress tracking.
        """
        page_layouts: List[PageLayout] = []

        if file_kind == "pdf":
            # Extract pages sequentially
            # Count total pages for progress callback if provided
            total_pages = None
            try:
                temp_doc = fitz.open(stream=file_bytes, filetype="pdf")
                total_pages = len(temp_doc)
                temp_doc.close()
            except Exception:
                pass

            for page_idx, img_bgr in self.iter_pdf_pages(file_bytes, dpi=dpi):
                work_img = img_bgr
                if page_transform_fn is not None:
                    work_img = page_transform_fn(page_idx, work_img)

                with self._inference_lock:
                    layout = self.reconstructor.process_page(work_img, page_index=page_idx, dpi=dpi)
                page_layouts.append(layout)
                if progress_callback and total_pages:
                    progress_callback(page_idx + 1, total_pages)
                del img_bgr
        elif file_kind == "image":
            img_bgr = self.decode_image_bytes(file_bytes)
            work_img = img_bgr
            if page_transform_fn is not None:
                work_img = page_transform_fn(0, work_img)

            with self._inference_lock:
                layout = self.reconstructor.process_page(work_img, page_index=0, dpi=dpi)
            page_layouts.append(layout)
            if progress_callback:
                progress_callback(1, 1)
            del img_bgr
        else:
            raise SecurityValidationError(f"Unsupported file kind: {file_kind}")

        return page_layouts

    def process_file_bytes(
        self,
        file_bytes: bytes,
        file_kind: str,
        output_docx_path: Union[str, Path],
        dpi: int = 150,
        page_transform_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[Path, List[PageLayout]]:
        """Process image or PDF bytes through reconstruction and build a multi-page DOCX.
        
        Streamlines memory consumption by iterating through pages sequentially.
        Note: DocumentReconstructor.process_page already internally performs deskewing,
        so raw rasterized images are passed directly without duplicate deskewing.
        """
        output_path = Path(output_docx_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page_layouts = self.process_to_layouts(
            file_bytes=file_bytes,
            file_kind=file_kind,
            dpi=dpi,
            page_transform_fn=page_transform_fn,
        )

        # Build and save single unified document using DocxBuilder.build_from_pages
        saved_path = self.docx_builder.build_from_pages(page_layouts, output_path=output_path)
        return Path(saved_path), page_layouts


