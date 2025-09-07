from processors.transcript_processors.base_processor import BaseProcessor


class SiliconValleyBankProcessor(BaseProcessor):
    start_marker = "Prepared Remarks"
    end_marker = "[Operator signoff]"

    def preprocess_text(self, text: str) -> str:
        """
        Cleaning specific to Silicon Valley Bank transcripts.

        Parameters:
        - text: raw text string

        Returns:
        - cleaned_text: cleaned text string
        """

        base_cleaned_text = super().preprocess_text(text)

        clean_lines = []
        lines = base_cleaned_text.split("\n")

        for line in lines:
            ## check if line contains '--' eg
            # Greg Becker -- President and Chief Executive Officer
            # Jon Arfstrom -- RBC Capital Markets -- Analyst
            if line and "--" in line:
                person_details = line.split("--")
                if self.is_person_name(person_details[0].strip()):
                    line = person_details[0].strip()

            if line == "Advertisement":
                continue

            clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines)
        return cleaned_text
