"""
"""
import os
import io
import time
import warnings
import gc
import logging
from typing import Dict, Any, Union, List
import numpy as np
import cv2
import pypdfium2 as pdfium

# Optimized environment configuration
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaddleOCRSequential:
    """Fully sequential OCR processing class (no Ray)"""

    def __init__(self, lang="es", quality_mode="fast"):
        """
        Initialize sequential OCR

        Args:
            lang: OCR language ("es", "en", etc.)
            quality_mode: "fast", "balanced", "quality"
        """
        from paddleocr import PaddleOCR

        # Configuration by quality mode
        if quality_mode == "fast":
            ocr_params = {
                "lang": lang,
                "use_angle_cls": False,
                "det_db_box_thresh": 0.7,
                "det_db_thresh": 0.3,
                "det_db_unclip_ratio": 1.5,
                "show_log": False
            }
            self.scale = 0.8
            self.max_h = 600
        elif quality_mode == "balanced":
            ocr_params = {
                "lang": lang,
                "use_angle_cls": False,
                "det_db_box_thresh": 0.6,
                "det_db_thresh": 0.3,
                "show_log": False
            }
            self.scale = 1.0
            self.max_h = 800
        else:  # quality
            ocr_params = {
                "lang": lang,
                "use_angle_cls": True,
                "det_db_box_thresh": 0.5,
                "show_log": False
            }
            self.scale = 1.2
            self.max_h = 960

        self.ocr = PaddleOCR(**ocr_params)
        self.quality_mode = quality_mode
        self.lang = lang

        logger.info(f"Sequential PaddleOCR initialized - Lang: {lang}, Mode: {quality_mode}")

    def _render_pdf_optimized(self, pdf_bytes: bytes) -> List[np.ndarray]:
        """Optimized rendering with memory management"""
        pages = []
        pdf = None

        try:
            pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))

            if len(pdf) == 0:
                logger.warning("Empty PDF or no pages")
                return []

            for i in range(len(pdf)):
                page = None
                bmp = None

                try:
                    page = pdf.get_page(i)

                    if page is None:
                        logger.warning(f"Could not load page {i + 1}")
                        continue

                    # Render with optimized scale
                    bmp = page.render(scale=self.scale, rotation=0)

                    if bmp is None:
                        logger.warning(f"Could not render page {i + 1}")
                        continue

                    arr = bmp.to_numpy()[:, :, :3]  # BGRA -> BGR

                    # Optimized preprocessing
                    arr = self._preprocess_image(arr)

                    if arr is not None and arr.size > 0:
                        pages.append(arr)
                    else:
                        logger.warning(f"Empty image on page {i + 1}")

                except Exception as e:
                    logger.error(f"Error processing page {i + 1}: {e}")
                    continue

                finally:
                    if page is not None:
                        try:
                            page.close()
                        except:
                            pass
                    if bmp is not None:
                        try:
                            del bmp
                        except:
                            pass

                # GC every 5 pages for large PDFs
                if i % 5 == 0:
                    gc.collect()

        except Exception as e:
            logger.error(f"General error rendering PDF: {e}")

        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except:
                    pass
            gc.collect()

        return pages

    def _preprocess_image(self, arr: np.ndarray) -> np.ndarray:
        """Optimized image preprocessing"""
        try:
            if arr is None or arr.size == 0:
                return None

            # Convert to grayscale in fast mode
            if self.quality_mode == "fast":
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # Resize if it exceeds the maximum height
            h, w = arr.shape[:2]
            if self.max_h and h > self.max_h:
                ratio = self.max_h / float(h)
                new_w = int(w * ratio)
                arr = cv2.resize(arr, (new_w, self.max_h), interpolation=cv2.INTER_AREA)

            # Contrast enhancement for quality mode
            if self.quality_mode == "quality":
                arr = cv2.convertScaleAbs(arr, alpha=1.1, beta=10)

            return arr

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return None

    def _extract_text_from_ocr_result(self, ocr_result) -> str:
        """Optimized text extraction with confidence filters"""
        try:
            if not ocr_result or not ocr_result[0]:
                return ""

            # Confidence thresholds by mode
            confidence_thresholds = {
                "fast": 0.5,
                "balanced": 0.6,
                "quality": 0.7
            }
            min_confidence = confidence_thresholds.get(self.quality_mode, 0.6)

            text_parts = []
            for detection in ocr_result[0]:
                try:
                    if len(detection) >= 2 and len(detection[1]) >= 2:
                        text, confidence = detection[1]
                        if isinstance(text, str) and confidence >= min_confidence:
                            text = text.strip()
                            if len(text) > 0:
                                text_parts.append(text)
                except Exception as e:
                    logger.debug(f"Error processing OCR detection: {e}")
                    continue

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from OCR result: {e}")
            return ""

    def process_pdf_content(self, pdf_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
        """
        Main function to process PDF content sequentially
        """
        start_time = time.time()

        # Initial validation
        if not pdf_bytes or len(pdf_bytes) == 0:
            return {
                "success": False,
                "error": "Empty or null PDF",
                "text": "",
                "filename": filename,
                "latency_ms": 0,
                "pages_processed": 0,
                "characters_extracted": 0
            }

        try:
            # Render PDF to images
            pages = self._render_pdf_optimized(pdf_bytes)

            if not pages:
                return {
                    "success": False,
                    "error": "Could not render pages from the PDF",
                    "text": "",
                    "filename": filename,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "pages_processed": 0,
                    "characters_extracted": 0
                }

            # Process each page with OCR
            page_texts = []
            pages_processed = 0

            for i, page_img in enumerate(pages):
                try:
                    if page_img is None or page_img.size == 0:
                        logger.warning(f"Page {i + 1} empty, skipping...")
                        continue

                    # Run OCR
                    ocr_result = self.ocr.ocr(page_img, cls=False)
                    page_text = self._extract_text_from_ocr_result(ocr_result)

                    if page_text.strip():
                        page_texts.append(f"=== Page {i + 1} ===")
                        page_texts.append(page_text.strip())
                        page_texts.append("")

                    pages_processed += 1

                except Exception as e:
                    logger.warning(f"OCR error on page {i + 1} of {filename}: {e}")
                    continue

            # Combine final text
            final_text = "\n".join(page_texts).strip()
            processing_time = time.time() - start_time

            # Determine success based on extracted content
            success = len(final_text) > 10 and pages_processed > 0

            return {
                "success": success,
                "text": final_text,
                "filename": filename,
                "pages_processed": pages_processed,
                "characters_extracted": len(final_text),
                "latency_ms": int(processing_time * 1000),
                "quality_mode": self.quality_mode,
                "lang": self.lang,
                "error": None if success else "Insufficient text extracted"
            }

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"General error processing {filename}: {str(e)}"
            logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "filename": filename,
                "latency_ms": int(error_time * 1000),
                "pages_processed": 0,
                "characters_extracted": 0,
                "quality_mode": self.quality_mode,
                "lang": self.lang
            }
        finally:
            # Final memory cleanup
            gc.collect()


def extract_text_sequential_pure(
        pdf_content: Union[bytes, str],
        filename: str = "unknown",
        lang: str = "es",
        quality_mode: str = "fast"
) -> Dict[str, Any]:
    """
    Main FULLY SEQUENTIAL text-extraction function (no Ray)

    Args:
        pdf_content: PDF content (bytes) or path to the file (str)
        filename: Filename for logging
        lang: OCR language ("es", "en", etc.)
        quality_mode: Quality mode ("fast", "balanced", "quality")

    Returns:
        Dict with the processing result
    """

    # Input validation
    if not pdf_content:
        return {
            "success": False,
            "error": "Empty PDF content",
            "text": "",
            "filename": filename,
            "latency_ms": 0
        }

    start_time = time.time()

    try:
        # Convert to bytes if it is a file path
        if isinstance(pdf_content, str):
            if os.path.isfile(pdf_content):
                try:
                    with open(pdf_content, 'rb') as f:
                        pdf_bytes = f.read()
                    filename = filename if filename != "unknown" else os.path.basename(pdf_content)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error reading file {pdf_content}: {str(e)}",
                        "text": "",
                        "filename": filename,
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
            else:
                return {
                    "success": False,
                    "error": f"File not found: {pdf_content}",
                    "text": "",
                    "filename": filename,
                    "latency_ms": int((time.time() - start_time) * 1000)
                }
        else:
            pdf_bytes = pdf_content

        logger.info(f"Processing {filename} in pure sequential mode")

        # Create the sequential processor
        processor = PaddleOCRSequential(
            lang=lang,
            quality_mode=quality_mode
        )

        # Process PDF
        result = processor.process_pdf_content(pdf_bytes, filename)

        # Add extra information
        result["total_latency_ms"] = int((time.time() - start_time) * 1000)
        result["cores_used"] = 1
        result["mode"] = "sequential_pure"

        logger.info(
            f"Pure sequential completed - {result.get('characters_extracted', 0)} characters in {result['total_latency_ms']}ms")

        return result

    except Exception as e:
        error_time = time.time() - start_time
        return {
            "success": False,
            "error": f"General error in sequential extraction: {str(e)}",
            "text": "",
            "filename": filename,
            "latency_ms": int(error_time * 1000),
            "exception": type(e).__name__,
            "cores_used": 1,
            "mode": "sequential_pure"
        }


# =========================
#   TEST FUNCTION
# =========================

def test_sequential_pure():
    """Test function for the pure sequential version"""

    # Create a dummy PDF for testing
    dummy_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000201 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
295
%%EOF"""

    # Test the sequential function
    result = extract_text_sequential_pure(
        pdf_content=dummy_pdf,
        filename="test_sequential.pdf",
        lang="es",
        quality_mode="fast"
    )

    print("Pure sequential result:")
    print(f"- Success: {result['success']}")
    print(f"- Extracted text: {len(result.get('text', ''))} characters")
    print(f"- Time: {result['total_latency_ms']}ms")
    print(f"- Mode: {result['mode']}")

    if not result['success']:
        print(f"- Error: {result['error']}")

    return result


if __name__ == "__main__":
    test_sequential_pure()
