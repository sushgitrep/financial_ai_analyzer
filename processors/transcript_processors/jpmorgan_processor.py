import re
from processors.transcript_processors.base_processor import BaseProcessor


class JPMorganProcessor(BaseProcessor):
    start_marker = "MANAGEMENT DISCUSSION SECTION"
    end_marker = "Disclaimer"

    def preprocess_text(self, text: str) -> str:
        """
        Cleaning specific to JPMorgan transcripts.

        Parameters:
        - text: raw text string

        Returns:
        - cleaned_text: cleaned text string
        """

        base_cleaned_text = super().preprocess_text(text)

        clean_lines = []
        lines = base_cleaned_text.split("\n")
        previous_line_was_name = False

        for line in lines:
            # line contains only multiple dots or empty spaces
            if re.match(r"^\.+$", line):
                continue

            # if line contains only 'A' or 'Q' (question/answer) remove it
            if line in ("A", "Q"):
                continue

            is_person_name = self.is_person_name(line)

            # if line is a person name next will be position - remove position
            if previous_line_was_name:
                previous_line_was_name = False
                continue

            # if line like Operator: bla bla split it in 2 lines without :
            if line.startswith("Operator:"):
                new_lines = line.split(":", 1)
                clean_lines.append(new_lines[0])
                clean_lines.append(new_lines[1].strip())
                previous_line_was_name = False
                continue

            clean_lines.append(line)

            previous_line_was_name = is_person_name

        cleaned_text = "\n".join(clean_lines)

        return cleaned_text
