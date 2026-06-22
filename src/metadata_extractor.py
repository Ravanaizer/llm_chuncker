# metadata_extractor.py
import re
from typing import Dict


def extract_metadata(text: str, filename: str) -> Dict[str, str]:
    metadata = {
        "filename": filename,
        "contract_number": "",
        "contract_date": "",
        "doc_type": "",
        "organization": "",
        "amount": "",
    }

    # Извлечение номера договора
    patterns = [
        r"договор[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"приказ[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"накладн[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"№\s*([\d./\-]+/20\d{2}[а-яА-Я]?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            metadata["contract_number"] = match.group(1).strip()
            break

    # Извлечение даты
    date_patterns = [
        r'[«"](\d{1,2})[»"]\s+([а-я]+)\s+(\d{4})\s*г',
        r"(\d{2}\.\d{2}\.\d{4})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            if len(match.groups()) == 3:
                metadata["contract_date"] = (
                    f"{match.group(1)} {match.group(2)} {match.group(3)}"
                )
            else:
                metadata["contract_date"] = match.group(1)
            break

    # Извлечение организации
    org_patterns = [
        r'(ООО\s+"[^"]+")',
        r'(ПАО\s+"[^"]+")',
        r'с\s+(ООО|ПАО|АО)\s+[«"]([^»"]+)[»"]',
    ]
    for pattern in org_patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                metadata["organization"] = f"{match.group(1)} «{match.group(2)}»"
            else:
                metadata["organization"] = match.group(0)
            break

    # Извлечение суммы
    amount_patterns = [
        r"сумм[а-я]*\s*[:\-]?\s*([\d\s.,]+)\s*(руб|₽)",
        r"([\d\s.,]+)\s*(руб|₽)",
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            metadata["amount"] = match.group(1).strip()
            break

    # Определение типа документа
    text_lower = text[:500].lower()
    if "договор" in text_lower:
        metadata["doc_type"] = "Договор"
    elif "приказ" in text_lower:
        metadata["doc_type"] = "Приказ"
    elif "служебная записка" in text_lower:
        metadata["doc_type"] = "Служебная записка"
    elif "акт" in text_lower:
        metadata["doc_type"] = "Акт"
    elif "накладная" in text_lower:
        metadata["doc_type"] = "Накладная"

    return metadata
