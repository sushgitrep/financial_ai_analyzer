from processors.transcript_processors.base_processor import BaseProcessor


class BankOfAmericaProcessor(BaseProcessor):
    header_margin = 40
    footer_margin = 30
    start_marker = "Presentation"
    end_marker = "Disclaimer"
