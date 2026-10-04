"""Normalization for business names, addresses and country labels.

Design rules
------------
*   Raw fields are never mutated.  Every function returns a NEW representation;
    callers keep `business_name` / `business_address` untouched alongside.
*   Nothing here consults any external resource.  The one non-obvious table --
    Indic -> Latin romanization -- is derived at import time from the Unicode
    character database that ships with Python (`unicodedata.name`), which is a
    character-encoding property, not business data.

Why romanization is mandatory here (measured, see artifacts/profile/profile.txt):
    ASCII-folding business_name the way the starter `normalize_text` does erases
    1,353,239 / 5,034,616 = 26.9% of Source-2 names and 7.4% of Source-3 names
    outright, because they are written in Devanagari, Telugu, Tamil, Kannada,
    Bengali, Gujarati, Malayalam, Oriya or Gurmukhi.  Source 1 is 100% Latin.
    Folding those to "" makes ~1.35M records permanently unmatchable on name.
"""
from __future__ import annotations

import re
import sys
import unicodedata

# ------------------------------------------------------------------------------
# Indic romanization, derived from the Unicode character database
# ------------------------------------------------------------------------------
# Indic blocks laid out on the common Brahmi-derived pattern.
_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
]

_INDEPENDENT_VOWELS = {
    "A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU",
    "VOCALIC R", "VOCALIC RR", "VOCALIC L", "VOCALIC LL",
    "CANDRA E", "CANDRA O", "CANDRA A", "SHORT E", "SHORT O", "SHORT A",
}

# Phoneme -> conventional ENGLISH spelling.  Keys are script-independent Unicode
# phoneme names, so this single table serves all ten blocks above.  The target is
# the spelling an English transliteration in Source 1 would actually use, because
# that is what we have to fuzzy-match against.
_PHONEME_TO_LATIN = {
    # vowels (long/short collapsed: matching benefits from fewer variants)
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u",
    "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au",
    "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li",
    "CANDRA E": "e", "CANDRA O": "o", "CANDRA A": "a",
    "SHORT E": "e", "SHORT O": "o", "SHORT A": "a",
    # velars
    "KA": "k", "KHA": "kh", "GA": "g", "GHA": "gh", "NGA": "ng",
    # palatals
    "CA": "ch", "CHA": "chh", "JA": "j", "JHA": "jh", "NYA": "ny",
    # retroflex (TT/DD/NN) folded onto the dental spelling English uses
    "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n",
    "TTTA": "t", "NNNA": "n", "RRA": "r", "LLLA": "l",
    # dentals
    "TA": "t", "THA": "th", "DA": "d", "DHA": "dh", "NA": "n",
    # labials
    "PA": "p", "PHA": "ph", "BA": "b", "BHA": "bh", "MA": "m",
    # semivowels / sibilants
    "YA": "y", "RA": "r", "LA": "l", "LLA": "l", "VA": "v",
    "SHA": "sh", "SSA": "sh", "SA": "s", "HA": "h",
    # Urdu/Persian extras used in Hindi
    "QA": "k", "KHHA": "kh", "GHHA": "g", "ZA": "z", "DDDHA": "r",
    "RHA": "rh", "FA": "f", "YYA": "y", "NNNNA": "n",
}

_ROMAN_LETTER: dict[str, tuple[str, bool]] = {}   # char -> (latin, is_consonant)
_ROMAN_VOWEL_SIGN: dict[str, str] = {}            # char -> latin
_VIRAMA: set[str] = set()
_NASAL: set[str] = set()                          # anusvara / candrabindu
_VISARGA: set[str] = set()
_SKIP: set[str] = set()                           # nukta, avagraha, length marks
_INDIC_DIGIT: dict[str, str] = {}

_DIGIT_NAMES = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
                "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}


def _build_indic_tables() -> None:
    for lo, hi in _INDIC_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            try:
                uname = unicodedata.name(ch)
            except ValueError:
                continue
            # names look like "<SCRIPT> LETTER KA", "<SCRIPT> VOWEL SIGN AA", ...
            parts = uname.split(" ", 1)
            if len(parts) != 2:
                continue
            rest = parts[1]
            if rest.startswith("LETTER "):
                ph = rest[len("LETTER "):]
                latin = _PHONEME_TO_LATIN.get(ph)
                if latin is None:
                    # unknown letter: keep its leading ascii letters as a guess
                    latin = re.sub(r"[^a-z]", "", ph.lower())[:2]
                is_cons = ph not in _INDEPENDENT_VOWELS
                _ROMAN_LETTER[ch] = (latin, is_cons)
            elif rest.startswith("VOWEL SIGN "):
                ph = rest[len("VOWEL SIGN "):]
                _ROMAN_VOWEL_SIGN[ch] = _PHONEME_TO_LATIN.get(
                    ph, re.sub(r"[^a-z]", "", ph.lower())[:2])
            elif rest.startswith("SIGN "):
                ph = rest[len("SIGN "):]
                if ph in ("VIRAMA", "PAMADI", "AL-LAKUNA", "HALANT"):
                    _VIRAMA.add(ch)
                elif ph in ("ANUSVARA", "CANDRABINDU", "ANUSVARA ABOVE"):
                    _NASAL.add(ch)
                elif ph == "VISARGA":
                    _VISARGA.add(ch)
                else:
                    _SKIP.add(ch)   # nukta, avagraha, ai/au length marks, etc.
            elif rest.startswith("DIGIT "):
                _INDIC_DIGIT[ch] = _DIGIT_NAMES.get(rest[len("DIGIT "):], "")
            elif rest.startswith("VOWEL "):   # Tamil/Sinhala independent vowels
                ph = rest[len("VOWEL "):]
                _ROMAN_LETTER[ch] = (_PHONEME_TO_LATIN.get(ph, ""), False)


_build_indic_tables()

_HAS_INDIC = re.compile("[" + "".join(f"\\u{lo:04x}-\\u{hi:04x}"
                                      for lo, hi in _INDIC_RANGES) + "]")


def romanize_indic(text: str) -> str:
    """Transliterate Indic-script runs to Latin; leave everything else alone.

    Consonants carry an inherent 'a' which is suppressed by a following virama or
    replaced by a following vowel sign -- the standard abugida rule.
    """
    if not text or not _HAS_INDIC.search(text):
        return text
    out: list[str] = []
    pending_cons = False           # a consonant awaiting its inherent 'a'
    for ch in text:
        if ch in _SKIP:
            continue
        ent = _ROMAN_LETTER.get(ch)
        if ent is not None:
            if pending_cons:
                out.append("a")
            latin, is_cons = ent
            out.append(latin)
            pending_cons = is_cons
            continue
        vs = _ROMAN_VOWEL_SIGN.get(ch)
        if vs is not None:
            out.append(vs)          # replaces the inherent 'a'
            pending_cons = False
            continue
        if ch in _VIRAMA:
            pending_cons = False    # inherent 'a' suppressed
            continue
        if ch in _NASAL:
            if pending_cons:
                out.append("a")
                pending_cons = False
            out.append("n")
            continue
        if ch in _VISARGA:
            if pending_cons:
                out.append("a")
                pending_cons = False
            out.append("h")
            continue
        d = _INDIC_DIGIT.get(ch)
        if d is not None:
            if pending_cons:
                out.append("a")
                pending_cons = False
            out.append(d)
            continue
        # non-Indic character
        if pending_cons:
            out.append("a")
            pending_cons = False
        out.append(ch)
    if pending_cons:
        out.append("a")
    return "".join(out)


# ------------------------------------------------------------------------------
# generic text normalization
# ------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")

# '&' -> ' and ' before punctuation stripping ("B & B" == "B and B"); likewise the
# French/Latin ligatures, which NFKD does not decompose.
_LIGATURES = str.maketrans({"æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
                           "ß": "ss", "ø": "o", "Ø": "o", "đ": "d", "ð": "d",
                           "þ": "th", "ł": "l", "Ł": "l"})


def normalize_unicode(text: str) -> str:
    """NFKC compatibility form, control characters removed, whitespace collapsed."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def fold_ascii(text: str) -> str:
    """Strip diacritics (Président -> President, Léarning -> Learning).

    Applied only AFTER romanize_indic, so Indic text has already become Latin and
    nothing is lost.  Ligatures are expanded first because NFKD leaves them intact.
    """
    if not text:
        return ""
    text = text.translate(_LIGATURES)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


_SINGLE_LETTER_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")


def _collapse_initials(text: str) -> str:
    """Join runs of single letters: 's a s' -> 'sas', 'l l c' -> 'llc'.

    Punctuation stripping turns 'S.A.S' and 'S.A.R.L' into spaced letters, which
    would otherwise never match the unpunctuated 'SAS' / 'SARL' spelling used
    elsewhere in the same data.
    """
    return _SINGLE_LETTER_RUN.sub(lambda m: m.group(0).replace(" ", ""), text)


def _base_clean(text: str) -> str:
    """unicode -> romanize -> '&'->and -> fold -> lowercase -> alnum -> initials."""
    text = normalize_unicode(text)
    if not text:
        return ""
    text = romanize_indic(text)
    text = text.replace("&", " and ")
    text = fold_ascii(text).lower()
    text = _NON_ALNUM.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return _collapse_initials(text)


def normalize_name(text: str) -> str:
    """Canonical comparable form of a business name."""
    return _base_clean(text)


# ------------------------------------------------------------------------------
# legal suffixes / corporate forms
# ------------------------------------------------------------------------------
# Tokens that carry (almost) no identifying information: they say what KIND of
# legal entity this is, not WHICH business.  Covering US, India and France because
# the test set is US + India + France; the list is applied as a token filter, so an
# unseen country simply contributes no removals (nothing breaks).
# SEED list only -- unambiguous legal-form words.  The list actually used at
# runtime is discovered from the training ground truth by src/discover_tokens.py
# (a token is droppable when matched pairs disagree about its presence far more
# often than a genuinely identifying token would) and cached to
# artifacts/profile/legal_tokens.json.  Deliberately NOT included here:
# "international", "group", "holdings", "trust", "enterprises" -- they look like
# suffixes but carry real identifying information.
LEGAL_TOKENS_SEED: set[str] = {
    # US / generic anglophone
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc",
    "llp", "lp", "plc", "ltd", "limited", "pllc", "pc",
    # India
    "pvt", "private", "pte", "opc",
    # India, romanized from Indic script by romanize_indic()
    "praivet", "praiveta", "limiteda", "elaelapi", "kampani",
    # France
    "sarl", "sa", "sas", "sasu", "sci", "eurl", "snc", "selarl", "sarlu",
    "scs", "eirl", "ei",
}


def _load_legal_tokens() -> set[str]:
    """Runtime legal-token set: data-derived file when present, else the seed."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "artifacts" / "profile" / "legal_tokens.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text())["legal_tokens"])
        except Exception:
            pass
    return set(LEGAL_TOKENS_SEED)


LEGAL_TOKENS: set[str] = _load_legal_tokens()

# Suffix phrases removed as whole trailing/leading strings (order matters).
LEGAL_PHRASES: list[str] = [
    "private limited", "private ltd", "pvt limited", "pvt ltd", "pvt ltd co",
    "limited liability company", "limited liability partnership",
    "and company", "and co", "and sons", "and brothers",
    "societe a responsabilite limitee", "societe par actions simplifiee",
    "societe civile immobiliere", "entreprise unipersonnelle",
    "praivet limiteda", "praiveta limiteda",
]


def strip_legal_suffix(name_norm: str) -> str:
    """Drop legal-form tokens from an already-normalized name.

    Never returns "" when that would destroy the record: if every token is a legal
    token (e.g. the name literally is "private limited"), the input is kept.
    """
    if not name_norm:
        return ""
    s = name_norm
    for ph in LEGAL_PHRASES:
        if s.endswith(" " + ph):
            s = s[: -len(ph) - 1]
        elif s.startswith(ph + " "):
            s = s[len(ph) + 1:]
    toks = [t for t in s.split() if t not in LEGAL_TOKENS]
    if not toks:
        return name_norm
    return " ".join(toks)


# ------------------------------------------------------------------------------
# addresses
# ------------------------------------------------------------------------------
# Canonical short form for street-type and unit words.  Both spellings map to the
# same key so "Road"/"Rd" and "Rue"/"R" agree.  Values are deliberately short.
ADDRESS_CANON: dict[str, str] = {
    # US street types
    "road": "rd", "rd": "rd", "street": "st", "str": "st", "st": "st",
    "avenue": "ave", "ave": "ave", "av": "ave", "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir", "place": "pl", "pl": "pl", "plaza": "plz",
    "plz": "plz", "square": "sq", "sq": "sq", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "expressway": "expy", "expy": "expy",
    "freeway": "fwy", "turnpike": "tpke", "tpke": "tpke", "route": "rte",
    "rte": "rte", "way": "way", "loop": "loop", "path": "path", "pike": "pike",
    "crossing": "xing", "xing": "xing", "point": "pt", "pt": "pt",
    "ridge": "rdg", "rdg": "rdg", "creek": "crk", "crk": "crk",
    "extension": "ext", "ext": "ext", "boulevar": "blvd",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "ne": "ne", "nw": "nw", "se": "se", "sw": "sw",
    # units
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "unit": "unit", "floor": "flr", "flr": "flr", "fl": "flr",
    "building": "bldg", "bldg": "bldg", "room": "rm", "rm": "rm",
    "number": "no", "no": "no", "num": "no", "door": "no",
    "post": "po", "po": "po", "box": "box", "block": "blk", "blk": "blk",
    "basement": "bsmt", "department": "dept", "dept": "dept",
    # India
    "nagar": "nagar", "colony": "colony", "cly": "colony", "sector": "sec",
    "sec": "sec", "phase": "ph", "ph": "ph", "opposite": "opp", "opp": "opp",
    "near": "near", "behind": "behind", "beside": "beside",
    "market": "mkt", "mkt": "mkt", "industrial": "indl", "indl": "indl",
    "estate": "estate", "vihar": "vihar", "puram": "puram", "pura": "puram",
    "gali": "gali", "galli": "gali", "marg": "marg", "path": "path",
    "chowk": "chowk", "chauk": "chowk", "bazar": "bazar", "bazaar": "bazar",
    "village": "vill", "vill": "vill", "taluk": "tq", "taluka": "tq", "tq": "tq",
    "district": "dist", "dist": "dist", "tehsil": "teh", "teh": "teh",
    "kh": "kh", "khasra": "kh", "khno": "kh", "plot": "plot", "gat": "gat",
    "society": "soc", "soc": "soc", "chs": "soc", "wadi": "wadi",
    "cross": "cross", "main": "main", "stage": "stage", "layout": "layout",
    "extn": "ext", "flat": "flat", "shop": "shop", "gr": "grnd",
    "ground": "grnd", "grnd": "grnd", "floorno": "flr",
    # France
    "rue": "rue", "r": "rue", "avenu": "ave", "boulevrd": "blvd",
    "allee": "all", "all": "all", "impasse": "imp", "imp": "imp",
    "chemin": "chem", "chem": "chem", "quai": "quai", "cours": "crs",
    "crs": "crs", "faubourg": "fbg", "fbg": "fbg", "residence": "res",
    "res": "res", "batiment": "bat", "bat": "bat", "etage": "flr",
    "appartement": "apt", "appt": "apt", "app": "apt", "lieu": "lieu",
    "lieudit": "lieu", "zone": "zone", "za": "za", "zi": "zi", "zac": "zac",
    "boite": "box", "bp": "box", "cedex": "cedex", "place": "pl",
    "passage": "pass", "pass": "pass", "sentier": "sent", "villa": "villa",
    "square": "sq", "rond": "rond", "pont": "pont", "port": "port",
}

# Words so common in addresses that they carry no discriminative power on their own.
ADDRESS_STOP: set[str] = {
    "rd", "st", "ave", "blvd", "dr", "ln", "ct", "cir", "pl", "sq", "way",
    "ste", "apt", "unit", "flr", "bldg", "rm", "no", "po", "box", "blk",
    "rue", "all", "imp", "chem", "res", "bat", "near", "opp", "behind",
    "beside", "and", "the", "of", "at", "in", "on", "de", "du", "des", "la",
    "le", "les", "a",
}


def normalize_address(text: str) -> str:
    """Canonical comparable form of an address, with abbreviations unified."""
    base = _base_clean(text)
    if not base:
        return ""
    toks = [ADDRESS_CANON.get(t, t) for t in base.split()]
    return " ".join(toks)


def normalize_country(text: str) -> str:
    """Normalize a country LABEL.

    Open set by construction: any unseen label (France, Brazil, ...) simply becomes
    its own lowercase token.  Nothing is filtered, mapped to a fixed vocabulary, or
    dropped, so an unseen country cannot break inference.
    """
    return _base_clean(text)


# ------------------------------------------------------------------------------
# tokenization / components
# ------------------------------------------------------------------------------
def tokenize(norm_text: str) -> list[str]:
    """Tokens of an already-normalized string."""
    return norm_text.split() if norm_text else []


_POSTCODE = re.compile(r"^\d{5,6}$")       # US ZIP (5) and India PIN / FR CP (5-6)
_HAS_DIGIT = re.compile(r"\d")


def extract_address_components(addr_norm: str) -> dict:
    """Pull structural pieces out of a normalized address using text alone.

    No geocoding, no gazetteer, no external table -- purely shape-based:
      postcode  : standalone 5-6 digit tokens
      numbers   : tokens containing a digit (house / plot / khasra numbers)
      words     : alphabetic tokens that are not street-type stopwords
      tail      : last two alphabetic tokens (city / state / region position in
                  all three countries' layouts observed in this dataset)
    """
    toks = addr_norm.split() if addr_norm else []
    postcodes, numbers, words = [], [], []
    for t in toks:
        if _POSTCODE.match(t):
            postcodes.append(t)
        elif _HAS_DIGIT.search(t):
            numbers.append(t)
        elif t not in ADDRESS_STOP:
            words.append(t)
    return {
        "postcode": postcodes,
        "numbers": numbers,
        "words": words,
        "tail": words[-2:] if words else [],
    }


# ------------------------------------------------------------------------------
def build_name_variants(raw_name: str) -> dict:
    """All name representations used downstream, from one raw name."""
    nn = normalize_name(raw_name)
    core = strip_legal_suffix(nn)
    toks = tokenize(core)
    return {
        "name_norm": nn,
        "name_core": core,
        "name_sorted": " ".join(sorted(toks)),
        "name_tokens": toks,
    }


if __name__ == "__main__":  # quick manual smoke test
    samples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "మॉडर्न फाइनेंस",
        "मॉडर्न फाइनेंस",
        '"""chrs & cie sasu"',
        "-- Holloway Peak Inc Seafood",
        "LLC Moncada Léarning Center",
        "Fractales Amis Groupe S.A.S",
        "SCI Ptit Àmicale",
        "B+ Retail Inc",
        "International South Consultants Private Ltd",
    ]
    for s in samples:
        v = build_name_variants(s)
        print(f"{s!r}\n   norm={v['name_norm']!r}\n   core={v['name_core']!r}")
    for a in ["175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
              "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
              "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR",
              "IA, Iowa City, 1064 Newton Rd, Unit 11"]:
        an = normalize_address(a)
        print(f"{a!r}\n   addr={an!r}\n   comp={extract_address_components(an)}")
    print("tables:", len(_ROMAN_LETTER), "letters,", len(_ROMAN_VOWEL_SIGN),
          "vowel signs,", len(_VIRAMA), "viramas", file=sys.stderr)
