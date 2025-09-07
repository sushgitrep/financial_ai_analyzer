import pdfplumber
import re
import fitz  # PyMuPDF


class BaseProcessor:
    header_margin = 20
    footer_margin = 20
    start_marker = None
    end_marker = None

    def __init__(self, bank_name: str):
        self.bank_name = bank_name

    def read_pdf_with_pdfplumber(self, file_path: str) -> str:
        """
        Extract text from PDF while removing headers and footers.

        Parameters:
        - pdf_path: path to the PDF file
        - header_margin: distance from top to ignore as header (points)
        - footer_margin: distance from bottom to ignore as footer (points)

        Returns:
        - text: string with combined text from all pages
        """
        full_text = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_height = page.height
                words = page.extract_words()

                # Filter out words in header/footer areas
                main_words = [
                    w
                    for w in words
                    if w["top"] > self.header_margin
                    and w["bottom"] < (page_height - self.footer_margin)
                ]

                # Group words into lines by 'top' coordinate
                lines = {}
                for w in main_words:
                    top_rounded = round(w["top"])
                    lines.setdefault(top_rounded, []).append(w["text"])

                # Sort lines by vertical position and join words
                page_text = "\n".join(
                    [" ".join(line_words) for top, line_words in sorted(lines.items())]
                )
                full_text.append(page_text)

        return "\n".join(full_text)

    def read_pdf_with_pymupdf(self, file_path: str) -> str:
        """
        Extract text from PDF while removing headers and footers using PyMuPDF.

        Parameters:
        - file_path: path to the PDF file

        Returns:
        - text: string with combined text from all pages
        """
        full_text = []

        doc = fitz.open(file_path)
        for page in doc:
            page_height = page.rect.height

            # Get words as (x0, y0, x1, y1, "word")
            words = page.get_text("words")

            # Filter out words in header/footer areas
            main_words = [
                w
                for w in words
                if w[1] > self.header_margin
                and w[3] < (page_height - self.footer_margin)
            ]

            # Group words into lines by rounded 'top' coordinate
            lines = {}
            for w in main_words:
                top_rounded = round(w[1])  # y0 coordinate
                lines.setdefault(top_rounded, []).append(w[4])  # the word text

            # Sort lines by vertical position and join words
            page_text = "\n".join(
                [" ".join(line_words) for top, line_words in sorted(lines.items())]
            )
            full_text.append(page_text)

        doc.close()
        return "\n".join(full_text)

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing.
        It returns text within start_marker and end_marker if they are defined.

        Parameters:
        - text: raw text string

        Returns:
        - cleaned_text: cleaned text string
        """
        clean_lines = []
        lines = text.split("\n")

        capturing = self.start_marker is None  # start immediately if no marker set
        for line in lines:
            # remove extra spaces and newlines
            line = line.strip()

            if not line:
                continue

            # if line contains only numbers remove it
            if re.match(r"^\d+$", line):
                continue

            if self.start_marker and line.startswith(self.start_marker):
                capturing = True
                continue  # don’t include the marker itself

            if self.end_marker and line.startswith(self.end_marker):
                capturing = False
                break  # stop once we hit the end marker

            if not capturing:
                continue

            clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines)
        return cleaned_text

    def is_person_name(self, line: str) -> bool:
        """
        Returns True if line looks like a person name in First Last format.
        Allows for middle names/initials.
        """

        # Remove extra spaces
        line = line.strip()

        # Regex: capitalized words, optional initials
        pattern = r"^[A-Z][a-zA-Z'’\-]+(?:\s(?:[A-Z]\.|[A-Z][a-zA-Z'’\-]+)){1,2}$"

        return re.match(pattern, line) is not None

    def split_text_into_sections(self, cleaned_text: str) -> list:
        """
        Splits cleaned text by speaker.
        If no speaker is detected, text is assigned to 'Unknown'.
        """
        sections = []
        current_speaker = "Unknown"
        current_text = []

        lines = cleaned_text.split("\n")

        for line in lines:
            # Speaker detection
            if self.is_person_name(line) or line == "Operator":
                if current_text:
                    sections.append(
                        {
                            "speaker": current_speaker,
                            "speech": " ".join(current_text).strip(),
                        }
                    )
                current_speaker = line
                current_text = []
            else:
                current_text.append(line)

        # Last block
        if current_text:
            sections.append(
                {"speaker": current_speaker, "speech": " ".join(current_text).strip()}
            )

        return sections
