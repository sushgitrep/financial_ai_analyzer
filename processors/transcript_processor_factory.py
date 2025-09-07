from processors.transcript_processors.base_processor import BaseProcessor
from processors.transcript_processors.jpmorgan_processor import JPMorganProcessor
from processors.transcript_processors.lehman_brothers_processor import (
    LehmanBrothersProcessor,
)
from processors.transcript_processors.bank_of_america_processor import (
    BankOfAmericaProcessor,
)
from processors.transcript_processors.silicon_valley_bank_processor import (
    SiliconValleyBankProcessor,
)


class TranscriptProcessorFactory:
    @staticmethod
    def create_processor(bank_name: str) -> BaseProcessor:
        bank_name_lower = bank_name.lower()
        if bank_name_lower == "jp_morgan":
            return JPMorganProcessor("JPMorgan")
        elif bank_name_lower == "lb":
            return LehmanBrothersProcessor("Lehman Brothers")
        elif bank_name_lower == "baml":
            return BankOfAmericaProcessor("Bank of America")
        elif bank_name_lower == "svb":
            return SiliconValleyBankProcessor("Silicon Valley Bank")
        else:
            raise ValueError(f"No processor available for bank: {bank_name}")
