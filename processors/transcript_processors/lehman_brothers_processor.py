import re
from processors.transcript_processors.base_processor import BaseProcessor


class LehmanBrothersProcessor(BaseProcessor):
    header_margin = 100
    footer_margin = 100

    start_marker = "PRESENTATION"
    end_marker = "DISCLAIMER"

    def preprocess_text(self, text: str) -> str:
        """
        Cleaning specific to Lehman Brothers transcripts.

        Parameters:
        - text: raw text string

        Returns:
        - cleaned_text: cleaned text string
        """

        base_cleaned_text = super().preprocess_text(text)

        clean_lines = []
        lines = base_cleaned_text.split("\n")

        for line in lines:
            ## check if line contains '-' eg
            # Ian Lowitt- Lehman Brothers- CFO
            # Glenn Schorr- UBS- Analys
            if line and "-" in line:
                person_details = line.split("-")
                person_name = person_details[0].strip()

                # hack for Ian Lowitt-Lehman Brothers-CFO
                # because it doesn't read I properly
                replacements = [
                    (r"\bIa\s*n Lowitt\b", "Ian Lowitt"),  # handles Ia n
                    (r"\bI an Lowitt\b", "Ian Lowitt"),
                    (r"\ban Lowitt\b", "Ian Lowitt"),
                    (r"\bI Lowitt\b", "Ian Lowitt"),
                    (r"\bLowitt\b", "Ian Lowitt"),  # fallback, last
                ]

                for pattern, replacement in replacements:
                    if re.search(pattern, person_name):
                        person_name = re.sub(pattern, replacement, person_name)
                        break
                if line == "I Brothers-CFO":
                    continue

                if self.is_person_name(person_name):
                    line = person_name

            clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines)

        return cleaned_text
