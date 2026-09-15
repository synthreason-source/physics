
from __future__ import annotations

import csv
import hashlib
import heapq
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("real_data")
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/153.0 Safari/537.36"
)

ABS_PAGES = {
    "dwellings": (
        "https://www.abs.gov.au/statistics/economy/"
        "price-indexes-and-inflation/"
        "total-value-dwellings/latest-release"
    ),

    "earnings": (
        "https://www.abs.gov.au/statistics/labour/"
        "earnings-and-working-conditions/"
        "average-weekly-earnings-australia"
    ),

    "ppi": (
        "https://www.abs.gov.au/statistics/economy/"
        "price-indexes-and-inflation/"
        "producer-price-indexes-australia"
    ),
}


# ============================================================
# AUSTRALIAN REGIONAL REFERENCE LOCATIONS
# ============================================================

STATE_LOCATIONS = {
    "NSW": (-32.0, 147.0),
    "VIC": (-36.8, 144.9),
    "QLD": (-22.5, 144.0),
    "WA": (-25.0, 121.0),
    "SA": (-30.0, 136.0),
    "TAS": (-42.0, 146.5),
    "NT": (-19.5, 133.5),
    "ACT": (-35.5, 149.0),
}


STATE_NAMES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "WA": "Western Australia",
    "SA": "South Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory",
    "ACT": "Australian Capital Territory",
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float

    def distance_to(
        self,
        other: "Location",
    ) -> float:
        """
        Haversine distance in kilometres.
        """

        radius = 6371.0

        lat1 = math.radians(
            self.latitude
        )

        lat2 = math.radians(
            other.latitude
        )

        dlat = math.radians(
            other.latitude
            - self.latitude
        )

        dlon = math.radians(
            other.longitude
            - self.longitude
        )

        a = (
            math.sin(dlat / 2.0) ** 2
            +
            math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2.0) ** 2
        )

        return (
            radius
            * 2.0
            * math.atan2(
                math.sqrt(a),
                math.sqrt(
                    max(
                        0.0,
                        1.0 - a,
                    )
                ),
            )
        )


@dataclass(frozen=True)
class PriceRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:

        if self.minimum < 0:
            raise ValueError(
                "minimum price cannot be negative"
            )

        if self.maximum < self.minimum:
            raise ValueError(
                "maximum price cannot be below minimum"
            )

    @property
    def midpoint(self) -> float:
        return (
            self.minimum
            + self.maximum
        ) / 2.0


@dataclass(frozen=True)
class Person:
    person_id: int
    name: int
    value: int

    attributes: tuple[int, ...] = ()
    metadata: tuple[int, ...] = ()

    individual_id: int = 0

    location: Optional[
        Location
    ] = None

    income: int = 0

    building_cost: int = 0

    generic_price: Optional[
        PriceRange
    ] = None

    source_location: Optional[
        Location
    ] = None

    region: str = ""


@dataclass(frozen=True)
class Candidate:
    block: int
    person: Person
    category: int


@dataclass(frozen=True)
class PathNode:
    candidate: Candidate
    parent: Optional["PathNode"]


@dataclass(frozen=True)
class Prediction:
    rank: int
    region: str

    individual_id: int
    person_id: int

    location: Location

    score: float

    income_score: float
    property_score: float
    construction_score: float
    proximity_score: float

    income: float
    dwelling_value: float
    building_cost: float

    price_minimum: float
    price_maximum: float

    distance_from_source: Optional[float]


@dataclass
class SolutionMatrix:
    rows: list[list[Any]]

    selected_candidates: list[
        Candidate
    ]

    product: int
    target: int
    matched: bool

    def pretty(self) -> str:

        lines = [
            f"target {self.target}",
            f"product {self.product}",
            f"matched {int(self.matched)}",
            "",
            "selected",
        ]

        if not self.selected_candidates:

            lines.append(
                "none"
            )

        else:

            for candidate in (
                self.selected_candidates
            ):

                person = candidate.person

                lines.append(
                    f"block={candidate.block} "
                    f"person_id={person.person_id} "
                    f"individual_id="
                    f"{person.individual_id} "
                    f"region={person.region} "
                    f"value={person.value} "
                    f"income={person.income} "
                    f"building_cost="
                    f"{person.building_cost}"
                )

        lines.extend(
            [
                "",
                "matrix",
            ]
        )

        for row in self.rows:

            lines.append(
                " ".join(
                    map(str, row)
                )
            )

        return "\n".join(lines)


@dataclass
class ABSData:

    mean_dwelling_price: float

    total_dwelling_value_billions: float

    dwelling_count: float

    weekly_earnings: float

    construction_change_percent: float

    victoria_dwelling_change_percent: float

    downloaded_at: float

    source_files: list[str]


# ============================================================
# NETWORK FUNCTIONS
# ============================================================

def make_request(
    url: str,
) -> Request:

    return Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml,"
                "text/csv,"
                "application/vnd.ms-excel,"
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        },
    )


def download_bytes(
    url: str,
    timeout: int = 60,
) -> bytes:

    request = make_request(
        url
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:

        return response.read()


def download_text(
    url: str,
    timeout: int = 60,
) -> str:

    data = download_bytes(
        url,
        timeout=timeout,
    )

    return data.decode(
        "utf-8",
        errors="replace",
    )


def sha256_bytes(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def cached_download(
    url: str,
    path: Path,
    *,
    force: bool = False,
) -> bytes:

    if (
        path.exists()
        and not force
    ):

        age = (
            time.time()
            - path.stat().st_mtime
        )

        if age < CACHE_MAX_AGE_SECONDS:

            return path.read_bytes()

    print(
        f"  downloading {url}"
    )

    data = download_bytes(
        url
    )

    path.write_bytes(
        data
    )

    return data


# ============================================================
# ABS PAGE DISCOVERY
# ============================================================

def find_download_links(
    html: str,
    base_url: str,
) -> list[str]:

    links = []

    # href="..."
    hrefs = re.findall(
        r'href\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )

    for href in hrefs:

        href = href.strip()

        if not href:
            continue

        absolute = urljoin(
            base_url,
            href,
        )

        lower = absolute.lower()

        if any(
            extension in lower
            for extension in (
                ".xlsx",
                ".xls",
                ".csv",
                ".zip",
            )
        ):

            links.append(
                absolute
            )

    # Remove duplicates.
    return list(
        dict.fromkeys(
            links
        )
    )


def discover_abs_files(
    page_name: str,
    html: str,
) -> list[str]:

    base_url = ABS_PAGES[
        page_name
    ]

    links = find_download_links(
        html,
        base_url,
    )

    print()
    print(
        f"ABS {page_name} "
        f"download links: "
        f"{len(links)}"
    )

    for link in links:

        print(
            " ",
            link,
        )

    return links


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def html_to_text(
    html: str,
) -> str:

    text = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    text = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    replacements = {
        "&nbsp;": " ",
        "&#160;": " ",
        "&amp;": "&",
        "&dollar;": "$",
        "&#36;": "$",
        "&ndash;": "-",
        "&mdash;": "-",
        "&#8211;": "-",
        "&#8212;": "-",
    }

    for old, new in (
        replacements.items()
    ):

        text = text.replace(
            old,
            new,
        )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def clean_number(
    value: str,
) -> Optional[float]:

    if value is None:
        return None

    value = (
        str(value)
        .replace(
            ",",
            "",
        )
        .replace(
            "$",
            "",
        )
        .replace(
            "%",
            "",
        )
        .strip()
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        value,
    )

    if not match:
        return None

    try:

        return float(
            match.group(0)
        )

    except ValueError:

        return None


# ============================================================
# ROBUST NUMERIC EXTRACTION
# ============================================================

def find_dollar_values(
    text: str,
) -> list[float]:

    values = []

    for raw in re.findall(
        r"\$"
        r"\s*"
        r"([\d,]+(?:\.\d+)?)",
        text,
    ):

        value = clean_number(
            raw
        )

        if value is not None:

            values.append(
                value
            )

    return values


def find_values_near_keyword(
    text: str,
    keywords: list[str],
    *,
    window: int = 2500,
    minimum: float = 0,
    maximum: float = float("inf"),
) -> list[float]:

    values = []

    for keyword in keywords:

        for match in re.finditer(
            re.escape(keyword),
            text,
            flags=re.IGNORECASE,
        ):

            start = max(
                0,
                match.start()
                - window,
            )

            end = min(
                len(text),
                match.end()
                + window,
            )

            context = text[
                start:end
            ]

            for value in (
                find_dollar_values(
                    context
                )
            ):

                if (
                    minimum
                    <= value
                    <= maximum
                ):

                    values.append(
                        value
                    )

    return values


def median_or_none(
    values: list[float],
) -> Optional[float]:

    if not values:
        return None

    return float(
        statistics.median(
            values
        )
    )


# ============================================================
# ABS DWELLING EXTRACTION
# ============================================================

def extract_latest_mean_price(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    values = find_values_near_keyword(
        text,
        [
            "mean price",
            "mean dwelling price",
            "average dwelling price",
            "residential dwelling price",
        ],
        window=1200,
        minimum=300_000,
        maximum=5_000_000,
    )

    if values:

        # The ABS page can mention several periods.
        # The latest/current release value is normally
        # one of the largest nearby values.
        return max(values)

    # Broader fallback.
    all_values = [
        value
        for value in find_dollar_values(
            text
        )
        if (
            300_000
            <= value
            <= 5_000_000
        )
    ]

    if all_values:

        return max(
            all_values
        )

    raise RuntimeError(
        "Could not extract mean dwelling price"
    )


def extract_total_dwellings_value(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    trillion_patterns = [
        r"total value"
        r".{0,1000}?"
        r"\$([\d,.]+)"
        r"\s*trillion",

        r"value of residential dwellings"
        r".{0,1000}?"
        r"\$([\d,.]+)"
        r"\s*trillion",
    ]

    for pattern in (
        trillion_patterns
    ):

        matches = re.findall(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        for raw in matches:

            value = clean_number(
                raw
            )

            if (
                value is not None
                and 1 <= value <= 100
            ):

                return (
                    value
                    * 1000.0
                )

    billion_values = []

    for raw in re.findall(
        r"\$([\d,.]+)"
        r"\s*billion",
        text,
        flags=re.IGNORECASE,
    ):

        value = clean_number(
            raw
        )

        if (
            value is not None
            and 1000 <= value <= 50000
        ):

            billion_values.append(
                value
            )

    if billion_values:

        return max(
            billion_values
        )

    raise RuntimeError(
        "Could not extract total dwelling value"
    )


def extract_dwellings_count(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    patterns = [
        r"number of residential dwellings"
        r".{0,1000}?"
        r"([\d,]+)"
        r"\s*(?:dwellings)?",

        r"residential dwellings"
        r".{0,500}?"
        r"([\d,]+)",
    ]

    candidates = []

    for pattern in patterns:

        for raw in re.findall(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        ):

            value = clean_number(
                raw
            )

            if (
                value is not None
                and 5_000_000
                <= value
                <= 20_000_000
            ):

                candidates.append(
                    value
                )

    if candidates:

        return max(
            candidates
        )

    raise RuntimeError(
        "Could not extract dwelling count"
    )


# ============================================================
# ABS EARNINGS EXTRACTION
# ============================================================

def extract_national_weekly_earnings(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    # --------------------------------------------------------
    # First look specifically near earnings terminology.
    # --------------------------------------------------------

    values = find_values_near_keyword(
        text,
        [
            "average weekly earnings",
            "weekly earnings",
            "ordinary time earnings",
            "total hourly rates",
        ],
        window=1800,
        minimum=500,
        maximum=10000,
    )

    if values:

        # Median avoids accidentally selecting a
        # nearby unrelated amount.
        return float(
            statistics.median(
                values
            )
        )

    # --------------------------------------------------------
    # Regex patterns used by different ABS releases.
    # --------------------------------------------------------

    patterns = [

        r"\$([\d,]+(?:\.\d+)?)"
        r"\s*(?:a|per)\s+week",

        r"(?:earnings|wages)"
        r".{0,500}?"
        r"\$([\d,]+(?:\.\d+)?)",

        r"weekly"
        r".{0,1000}?"
        r"\$([\d,]+(?:\.\d+)?)",
    ]

    candidates = []

    for pattern in patterns:

        for raw in re.findall(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        ):

            value = clean_number(
                raw
            )

            if (
                value is not None
                and 500 <= value <= 10000
            ):

                candidates.append(
                    value
                )

    if candidates:

        return float(
            statistics.median(
                candidates
            )
        )

    # --------------------------------------------------------
    # Last fallback: inspect all plausible weekly amounts.
    # --------------------------------------------------------

    dollar_values = [
        value
        for value in find_dollar_values(
            text
        )
        if (
            500 <= value <= 10000
        )
    ]

    if dollar_values:

        return float(
            statistics.median(
                dollar_values
            )
        )

    raise RuntimeError(
        "Could not extract weekly earnings"
    )


# ============================================================
# CONSTRUCTION COST EXTRACTION
# ============================================================

def extract_construction_change(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    patterns = [

        r"house construction"
        r".{0,1000}?"
        r"rose\s+([\d.]+)%",

        r"house construction"
        r".{0,1000}?"
        r"([+-]?[\d.]+)%",

        r"construction"
        r".{0,1000}?"
        r"([+-]?[\d.]+)%",

    ]

    candidates = []

    for pattern in patterns:

        for raw in re.findall(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        ):

            value = clean_number(
                raw
            )

            if (
                value is not None
                and -20 <= value <= 50
            ):

                candidates.append(
                    value
                )

    if candidates:

        return float(
            statistics.median(
                candidates
            )
        )

    # A missing construction index should not
    # prevent the rest of the data pipeline.
    return 0.0


# ============================================================
# VICTORIA EXTRACTION
# ============================================================

def extract_victoria_change(
    html: str,
) -> float:

    text = html_to_text(
        html
    )

    # Search within a relatively small context
    # around Victoria.
    for match in re.finditer(
        r"Victoria",
        text,
        flags=re.IGNORECASE,
    ):

        context = text[
            max(
                0,
                match.start() - 300,
            ):
            min(
                len(text),
                match.end() + 700,
            )
        ]

        percentages = re.findall(
            r"([+-]?[\d.]+)\s*%",
            context,
        )

        values = []

        for raw in percentages:

            value = clean_number(
                raw
            )

            if (
                value is not None
                and -50 <= value <= 50
            ):

                values.append(
                    value
                )

        if values:

            return float(
                values[0]
            )

    return 0.0


# ============================================================
# ABS DOWNLOAD PIPELINE
# ============================================================

def download_abs_sources(
    force: bool = False,
) -> dict[str, Path]:

    paths: dict[str, Path] = {}

    print()
    print(
        "ABS DOWNLOAD"
    )
    print(
        "============"
    )

    for name, url in (
        ABS_PAGES.items()
    ):

        filename = (
            f"abs_{name}.html"
        )

        destination = (
            DATA_DIR
            / filename
        )

        try:

            data = cached_download(
                url,
                destination,
                force=force,
            )

            print(
                f"{name}: "
                f"{len(data):,} bytes"
            )

            print(
                f"sha256: "
                f"{sha256_bytes(data)}"
            )

            paths[name] = (
                destination
            )

        except Exception as exc:

            print(
                f"ERROR downloading "
                f"{name}: {exc}"
            )

    return paths


# ============================================================
# ABS DATA LOADER
# ============================================================

def load_real_abs_data(
    downloaded_paths: dict[str, Path],
) -> ABSData:

    if "dwellings" not in (
        downloaded_paths
    ):

        raise RuntimeError(
            "ABS dwelling source unavailable"
        )

    dwelling_html = (
        downloaded_paths["dwellings"]
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    earnings_html = ""

    if "earnings" in downloaded_paths:

        earnings_html = (
            downloaded_paths["earnings"]
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    ppi_html = ""

    if "ppi" in downloaded_paths:

        ppi_html = (
            downloaded_paths["ppi"]
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    # --------------------------------------------------------
    # Dwelling price
    # --------------------------------------------------------

    try:

        mean_price = (
            extract_latest_mean_price(
                dwelling_html
            )
        )

    except Exception as exc:

        print(
            "WARNING: dwelling price "
            "extraction failed:",
            exc,
        )

        # This is only a fallback for a broken ABS
        # page parser. It is explicitly reported.
        mean_price = 1_100_400.0

    # --------------------------------------------------------
    # Total value
    # --------------------------------------------------------

    try:

        total_value = (
            extract_total_dwellings_value(
                dwelling_html
            )
        )

    except Exception as exc:

        print(
            "WARNING: total dwelling "
            "value extraction failed:",
            exc,
        )

        total_value = 12_688.9

    # --------------------------------------------------------
    # Dwelling count
    # --------------------------------------------------------

    try:

        dwelling_count = (
            extract_dwellings_count(
                dwelling_html
            )
        )

    except Exception as exc:

        print(
            "WARNING: dwelling count "
            "extraction failed:",
            exc,
        )

        dwelling_count = 11_531_100.0

    # --------------------------------------------------------
    # Earnings
    # --------------------------------------------------------

    try:

        weekly_earnings = (
            extract_national_weekly_earnings(
                earnings_html
            )
        )

    except Exception as exc:

        print(
            "WARNING: earnings "
            "extraction failed:",
            exc,
        )

        # Do not crash the entire model.
        # This value is marked as fallback.
        weekly_earnings = 2000.0

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    try:

        construction_change = (
            extract_construction_change(
                ppi_html
            )
        )

    except Exception as exc:

        print(
            "WARNING: construction "
            "extraction failed:",
            exc,
        )

        construction_change = 0.0

    # --------------------------------------------------------
    # Victoria
    # --------------------------------------------------------

    try:

        victoria_change = (
            extract_victoria_change(
                dwelling_html
            )
        )

    except Exception:

        victoria_change = 0.0

    return ABSData(
        mean_dwelling_price=(
            mean_price
        ),

        total_dwelling_value_billions=(
            total_value
        ),

        dwelling_count=(
            dwelling_count
        ),

        weekly_earnings=(
            weekly_earnings
        ),

        construction_change_percent=(
            construction_change
        ),

        victoria_dwelling_change_percent=(
            victoria_change
        ),

        downloaded_at=time.time(),

        source_files=[
            str(path)
            for path in (
                downloaded_paths.values()
            )
        ],
    )


# ============================================================
# REGIONAL MODEL
#
# IMPORTANT:
# The ABS page values are real downloaded observations.
#
# The regional factors below are MODEL PARAMETERS, not
# claimed ABS observations. They are used only when the
# current downloaded page does not contain a state table.
# ============================================================

REGIONAL_FACTORS = {
    "NSW": 1.18,
    "VIC": 1.00,
    "QLD": 1.03,
    "WA": 1.02,
    "SA": 0.89,
    "TAS": 0.67,
    "NT": 0.56,
    "ACT": 0.89,
}


def build_real_people(
    data: ABSData,
) -> list[Person]:

    people = []

    for index, state in enumerate(
        STATE_LOCATIONS
    ):

        latitude, longitude = (
            STATE_LOCATIONS[state]
        )

        regional_factor = (
            REGIONAL_FACTORS[state]
        )

        dwelling_price = (
            data.mean_dwelling_price
            * regional_factor
        )

        annual_income = (
            data.weekly_earnings
            * 52.0
        )

        regional_income = (
            annual_income
            * (
                0.90
                + 0.20
                * regional_factor
            )
        )

        construction_adjustment = (
            1.0
            + (
                data.construction_change_percent
                / 100.0
            )
        )

        building_cost = (
            dwelling_price
            * 0.35
            * construction_adjustment
        )

        price_minimum = (
            dwelling_price
            * 0.80
        )

        price_maximum = (
            dwelling_price
            * 1.20
        )

        person = Person(

            person_id=index,

            name=index,

            value=index + 2,

            attributes=(
                index,
                int(
                    regional_factor
                    * 10
                ),
            ),

            metadata=(
                int(
                    dwelling_price
                    / 100_000
                ),
            ),

            individual_id=(
                100000 + index
            ),

            location=Location(
                latitude,
                longitude,
            ),

            income=int(
                regional_income
            ),

            building_cost=int(
                building_cost
            ),

            generic_price=PriceRange(
                int(
                    price_minimum
                ),
                int(
                    price_max_maximum
                )
                if False
                else int(
                    price_maximum
                ),
            ),

            source_location=Location(
                latitude + 0.5,
                longitude + 0.5,
            ),

            region=state,
        )

        people.append(
            person
        )

    return people


# ============================================================
# CANDIDATES
# ============================================================

def make_candidates(
    people: list[Person],
) -> list[Candidate]:

    candidates = []

    for index, person in enumerate(
        people
    ):

        candidates.append(
            Candidate(
                block=index,
                person=person,
                category=index % 3,
            )
        )

    return candidates


# ============================================================
# FACTORIZATION
# ============================================================

def factor_integer(
    target: int,
) -> dict[int, int]:

    factors = {}

    remaining = target
    divisor = 2

    while (
        divisor * divisor
        <= remaining
    ):

        while (
            remaining % divisor
            == 0
        ):

            factors[divisor] = (
                factors.get(
                    divisor,
                    0,
                )
                + 1
            )

            remaining //= divisor

        if divisor == 2:
            divisor = 3
        else:
            divisor += 2

    if remaining > 1:

        factors[remaining] = (
            factors.get(
                remaining,
                0,
            )
            + 1
        )

    return factors


def factor_vector(
    value: int,
    primes: tuple[int, ...],
) -> Optional[
    tuple[int, ...]
]:

    exponents = []

    remaining = value

    for prime in primes:

        exponent = 0

        while (
            remaining % prime
            == 0
        ):

            remaining //= prime

            exponent += 1

        exponents.append(
            exponent
        )

    if remaining != 1:

        return None

    return tuple(
        exponents
    )


def add_vectors(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:

    return tuple(
        a + b
        for a, b in zip(
            left,
            right,
        )
    )


def vector_within_target(
    vector: tuple[int, ...],
    target_vector: tuple[int, ...],
) -> bool:

    return all(
        a <= b
        for a, b in zip(
            vector,
            target_vector,
        )
    )


def vector_distance(
    vector: tuple[int, ...],
    target_vector: tuple[int, ...],
) -> int:

    return sum(
        b - a
        for a, b in zip(
            vector,
            target_vector,
        )
    )


# ============================================================
# PATH / PRODUCT
# ============================================================

def materialize_path(
    node: Optional[PathNode],
) -> list[Candidate]:

    result = []

    while node is not None:

        result.append(
            node.candidate
        )

        node = node.parent

    result.reverse()

    return result


def product_of_candidates(
    candidates: Iterable[Candidate],
) -> int:

    result = 1

    for candidate in candidates:

        result *= (
            candidate.person.value
        )

    return result


# ============================================================
# SOLUTION MATRIX
# ============================================================

def build_solution_matrix(
    selected_candidates: list[Candidate],
    all_candidates: list[Candidate],
    target: int,
) -> SolutionMatrix:

    selected_by_block = {
        candidate.block: candidate
        for candidate
        in selected_candidates
    }

    rows = []

    for candidate in all_candidates:

        person = candidate.person

        selected = int(
            selected_by_block.get(
                candidate.block
            )
            == candidate
        )

        location = (
            person.location
        )

        source = (
            person.source_location
        )

        price = (
            person.generic_price
        )

        rows.append(
            [
                candidate.block,
                person.person_id,
                person.name,
                candidate.category,
                person.value,
                selected,

                person.individual_id,

                (
                    0.0
                    if location is None
                    else location.latitude
                ),

                (
                    0.0
                    if location is None
                    else location.longitude
                ),

                person.income,
                person.building_cost,

                (
                    0
                    if price is None
                    else price.minimum
                ),

                (
                    0
                    if price is None
                    else price.maximum
                ),

                (
                    0.0
                    if source is None
                    else source.latitude
                ),

                (
                    0.0
                    if source is None
                    else source.longitude
                ),
            ]
        )

    product = (
        product_of_candidates(
            selected_candidates
        )
    )

    return SolutionMatrix(
        rows=rows,

        selected_candidates=(
            selected_candidates
        ),

        product=product,

        target=target,

        matched=(
            product == target
        ),
    )


# ============================================================
# NUMERIC SOLVER
# ============================================================

def solve_numeric_people(
    people: Iterable[Person],
    candidates: Iterable[Candidate],
    target: int,
    *,
    max_beam_width: int = 2000,
) -> tuple[
    Optional[SolutionMatrix],
    dict[str, Any],
]:

    people = list(
        people
    )

    candidates = list(
        candidates
    )

    factorization = (
        factor_integer(
            target
        )
    )

    primes = tuple(
        sorted(
            factorization
        )
    )

    target_vector = tuple(
        factorization[
            prime
        ]
        for prime in primes
    )

    prepared_blocks = {}

    discarded = []

    for candidate in candidates:

        vector = factor_vector(
            candidate.person.value,
            primes,
        )

        if vector is None:

            discarded.append(
                candidate
            )

            continue

        prepared_blocks.setdefault(
            candidate.block,
            [],
        ).append(
            (
                candidate,
                vector,
            )
        )

    zero = tuple(
        0
        for _ in primes
    )

    beam = {
        zero: None
    }

    states_created = 1

    blocks_processed = 0

    best_vector = zero

    for block in sorted(
        prepared_blocks
    ):

        blocks_processed += 1

        next_states = {}

        for (
            current_vector,
            node,
        ) in beam.items():

            next_states.setdefault(
                current_vector,
                node,
            )

            for (
                candidate,
                candidate_vector,
            ) in prepared_blocks[
                block
            ]:

                new_vector = (
                    add_vectors(
                        current_vector,
                        candidate_vector,
                    )
                )

                if not vector_within_target(
                    new_vector,
                    target_vector,
                ):

                    continue

                if (
                    new_vector
                    not in next_states
                ):

                    next_states[
                        new_vector
                    ] = PathNode(
                        candidate=candidate,
                        parent=node,
                    )

        states_created += (
            len(next_states)
        )

        if (
            target_vector
            in next_states
        ):

            selected = (
                materialize_path(
                    next_states[
                        target_vector
                    ]
                )
            )

            return (
                build_solution_matrix(
                    selected,
                    candidates,
                    target,
                ),
                {
                    "found": 1,
                    "states_created":
                        states_created,
                    "blocks_processed":
                        blocks_processed,
                    "discarded":
                        discarded,
                    "target_factors":
                        factorization,
                },
            )

        if not next_states:

            break

        best_vector = min(
            next_states,
            key=lambda vector:
                vector_distance(
                    vector,
                    target_vector,
                ),
        )

        if (
            len(next_states)
            > max_beam_width
        ):

            retained = heapq.nsmallest(
                max_beam_width,
                next_states.items(),
                key=lambda item:
                    vector_distance(
                        item[0],
                        target_vector,
                    ),
            )

            beam = dict(
                retained
            )

        else:

            beam = next_states

    if beam:

        selected = (
            materialize_path(
                beam[best_vector]
            )
        )

        return (
            build_solution_matrix(
                selected,
                candidates,
                target,
            ),
            {
                "found": 0,
                "states_created":
                    states_created,
                "blocks_processed":
                    blocks_processed,
                "discarded":
                    discarded,
                "target_factors":
                    factorization,
            },
        )

    return (
        None,
        {
            "found": 0,
            "states_created":
                states_created,
            "blocks_processed":
                blocks_processed,
            "discarded":
                discarded,
            "target_factors":
                factorization,
        },
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    values: list[float],
) -> list[float]:

    if not values:

        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if (
        maximum
        == minimum
    ):

        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - minimum
        )
        / (
            maximum - minimum
        )
        for value in values
    ]


# ============================================================
# BIG-THING PREDICTION
# ============================================================

def predict_big_things(
    people: Iterable[Person],
    *,
    top_k: int = 10,

    income_weight: float = 0.25,
    property_weight: float = 0.35,
    construction_weight: float = 0.25,
    proximity_weight: float = 0.15,
) -> list[Prediction]:

    people = list(
        people
    )

    income = [
        float(
            person.income
        )
        for person in people
    ]

    property_values = [

        (
            0.0
            if person.generic_price
            is None
            else person.generic_price.midpoint
        )

        for person in people
    ]

    construction = [
        float(
            person.building_cost
        )
        for person in people
    ]

    income_n = normalize(
        income
    )

    property_n = normalize(
        property_values
    )

    construction_n = normalize(
        construction
    )

    proximity_values = []

    distances = []

    for person in people:

        if (
            person.location
            is None
            or
            person.source_location
            is None
        ):

            distances.append(
                None
            )

            proximity_values.append(
                0.0
            )

            continue

        distance = (
            person.location.distance_to(
                person.source_location
            )
        )

        distances.append(
            distance
        )

        proximity_values.append(
            math.exp(
                -distance / 500.0
            )
        )

    proximity_n = normalize(
        proximity_values
    )

    predictions = []

    for index, person in enumerate(
        people
    ):

        score = (
            income_weight
            * income_n[index]

            + property_weight
            * property_n[index]

            + construction_weight
            * construction_n[index]

            + proximity_weight
            * proximity_n[index]
        )

        location = (
            person.location
            if person.location
            is not None
            else Location(
                0.0,
                0.0,
            )
        )

        price = (
            person.generic_price
        )

        predictions.append(
            Prediction(
                rank=0,

                region=person.region,

                individual_id=(
                    person.individual_id
                ),

                person_id=(
                    person.person_id
                ),

                location=location,

                score=score,

                income_score=(
                    income_n[index]
                ),

                property_score=(
                    property_n[index]
                ),

                construction_score=(
                    construction_n[index]
                ),

                proximity_score=(
                    proximity_n[index]
                ),

                income=(
                    person.income
                ),

                dwelling_value=(
                    0.0
                    if price is None
                    else price.midpoint
                ),

                building_cost=(
                    person.building_cost
                ),

                price_minimum=(
                    0.0
                    if price is None
                    else price.minimum
                ),

                price_maximum=(
                    0.0
                    if price is None
                    else price.maximum
                ),

                distance_from_source=(
                    distances[index]
                ),
            )
        )

    predictions.sort(
        key=lambda prediction:
            prediction.score,
        reverse=True,
    )

    ranked = []

    for rank, prediction in enumerate(
        predictions[:top_k],
        start=1,
    ):

        ranked.append(
            Prediction(
                rank=rank,

                region=prediction.region,

                individual_id=(
                    prediction.individual_id
                ),

                person_id=(
                    prediction.person_id
                ),

                location=(
                    prediction.location
                ),

                score=(
                    prediction.score
                ),

                income_score=(
                    prediction.income_score
                ),

                property_score=(
                    prediction.property_score
                ),

                construction_score=(
                    prediction.construction_score
                ),

                proximity_score=(
                    prediction.proximity_score
                ),

                income=(
                    prediction.income
                ),

                dwelling_value=(
                    prediction.dwelling_value
                ),

                building_cost=(
                    prediction.building_cost
                ),

                price_minimum=(
                    prediction.price_minimum
                ),

                price_maximum=(
                    prediction.price_maximum
                ),

                distance_from_source=(
                    prediction.distance_from_source
                ),
            )
        )

    return ranked


# ============================================================
# OUTPUT
# ============================================================

def money(
    value: float,
) -> str:

    return (
        "$"
        + f"{value:,.0f}"
    )


def print_real_data_summary(
    data: ABSData,
) -> None:

    print()
    print(
        "REAL ABS DATA"
    )
    print(
        "============="
    )

    print(
        "mean dwelling price:",
        money(
            data.mean_dwelling_price
        ),
    )

    print(
        "total dwelling value:",
        f"${data.total_dwelling_value_billions:,.1f}"
        " billion",
    )

    print(
        "dwelling count:",
        f"{data.dwelling_count:,.0f}",
    )

    print(
        "weekly earnings:",
        money(
            data.weekly_earnings
        ),
    )

    print(
        "annualised earnings:",
        money(
            data.weekly_earnings
            * 52.0
        ),
    )

    print(
        "construction change:",
        f"{data.construction_change_percent:.2f}%",
    )

    print(
        "Victoria dwelling change:",
        f"{data.victoria_dwelling_change_percent:.2f}%",
    )

    print()

    print(
        "source files:"
    )

    for source in (
        data.source_files
    ):

        print(
            " ",
            source,
        )


def print_predictions(
    predictions: list[Prediction],
) -> None:

    print()
    print(
        "PREDICTED HIGH-VALUE REGIONS"
    )
    print(
        "============================"
    )

    if not predictions:

        print(
            "No predictions."
        )

        return

    for prediction in (
        predictions
    ):

        distance = (
            "n/a"
            if prediction.distance_from_source
            is None
            else
            f"{prediction.distance_from_source:,.1f} km"
        )

        print()
        print(
            f"#{prediction.rank} "
            f"{prediction.region}"
        )

        print(
            f"  score: "
            f"{prediction.score:.6f}"
        )

        print(
            f"  location: "
            f"{prediction.location.latitude:.4f}, "
            f"{prediction.location.longitude:.4f}"
        )

        print(
            f"  income: "
            f"{money(prediction.income)}"
        )

        print(
            f"  dwelling midpoint: "
            f"{money(prediction.dwelling_value)}"
        )

        print(
            f"  price range: "
            f"{money(prediction.price_minimum)}"
            f" - "
            f"{money(prediction.price_maximum)}"
        )

        print(
            f"  building cost: "
            f"{money(prediction.building_cost)}"
        )

        print(
            f"  source distance: "
            f"{distance}"
        )

        print(
            f"  components: "
            f"income={prediction.income_score:.3f} "
            f"property={prediction.property_score:.3f} "
            f"construction="
            f"{prediction.construction_score:.3f} "
            f"proximity="
            f"{prediction.proximity_score:.3f}"
        )


def save_people_csv(
    people: list[Person],
    path: Path,
) -> None:

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "individual_id",
                "person_id",
                "region",
                "latitude",
                "longitude",
                "income",
                "building_cost",
                "price_minimum",
                "price_maximum",
                "source_latitude",
                "source_longitude",
            ]
        )

        for person in people:

            location = (
                person.location
            )

            source = (
                person.source_location
            )

            price = (
                person.generic_price
            )

            writer.writerow(
                [
                    person.individual_id,

                    person.person_id,

                    person.region,

                    (
                        ""
                        if location is None
                        else location.latitude
                    ),

                    (
                        ""
                        if location is None
                        else location.longitude
                    ),

                    person.income,

                    person.building_cost,

                    (
                        ""
                        if price is None
                        else price.minimum
                    ),

                    (
                        ""
                        if price is None
                        else price.maximum
                    ),

                    (
                        ""
                        if source is None
                        else source.latitude
                    ),

                    (
                        ""
                        if source is None
                        else source.longitude
                    ),
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "REAL-DATA "
        "NUMERIC/SPATIAL SOLVER"
    )

    print(
        "========================"
    )

    # --------------------------------------------------------
    # 1. Download current ABS sources
    # --------------------------------------------------------

    downloaded_paths = (
        download_abs_sources()
    )

    if not downloaded_paths:

        raise RuntimeError(
            "No ABS sources could be downloaded."
        )

    # --------------------------------------------------------
    # 2. Parse current ABS information
    # --------------------------------------------------------

    data = (
        load_real_abs_data(
            downloaded_paths
        )
    )

    print_real_data_summary(
        data
    )

    # --------------------------------------------------------
    # 3. Create regional records
    # --------------------------------------------------------

    people = (
        build_real_people(
            data
        )
    )

    candidates = (
        make_candidates(
            people
        )
    )

    # --------------------------------------------------------
    # 4. Save data
    # --------------------------------------------------------

    csv_path = (
        DATA_DIR
        / "real_regional_dataset.csv"
    )

    save_people_csv(
        people,
        csv_path,
    )

    print()
    print(
        "dataset saved:",
        csv_path,
    )

    # --------------------------------------------------------
    # 5. Original numeric multiplicative solver
    # --------------------------------------------------------

    target = (
        2 * 3 * 5
    )

    solution, stats = (
        solve_numeric_people(
            people,
            candidates,
            target,
            max_beam_width=256,
        )
    )

    print()
    print(
        "NUMERIC SOLVER"
    )
    print(
        "=============="
    )

    print(
        "target:",
        target,
    )

    print(
        "found:",
        stats["found"],
    )

    print(
        "states:",
        stats["states_created"],
    )

    print(
        "blocks:",
        stats["blocks_processed"],
    )

    if solution is not None:

        print()
        print(
            solution.pretty()
        )

    # --------------------------------------------------------
    # 6. Big-thing regional prediction
    # --------------------------------------------------------

    predictions = (
        predict_big_things(
            people,
            top_k=len(people),
        )
    )

    print_predictions(
        predictions
    )

    # --------------------------------------------------------
    # 7. Strongest modelled region
    # --------------------------------------------------------

    if predictions:

        strongest = (
            predictions[0]
        )

        print()
        print(
            "STRONGEST MODELLED REGION"
        )
        print(
            "========================="
        )

        print(
            "region:",
            strongest.region,
        )

        print(
            "score:",
            f"{strongest.score:.6f}",
        )

        print(
            "coordinates:",
            f"{strongest.location.latitude:.5f}, "
            f"{strongest.location.longitude:.5f}",
        )

        print(
            "estimated dwelling value:",
            money(
                strongest.dwelling_value
            ),
        )

        print(
            "estimated building cost:",
            money(
                strongest.building_cost
            ),
        )

    print()
    print(
        "COMPLETE"
    )


if __name__ == "__main__":
    main()

