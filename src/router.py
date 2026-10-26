"""Query router: picks a retrieval strategy before any retrieval happens,
so cheap/obvious queries (greetings, arithmetic, exact-match lookups) skip
the expensive paths entirely.

Rule order matters -- first match wins. Direct rules run first (cheapest,
most specific), then keyword rules, and anything left over falls through
to hybrid. All patterns are named constants here; see DECISIONS.md for why
each one is shaped the way it is.
"""

import re

# --- direct route patterns --------------------------------------------------

# Matches a greeting/thanks at the start of the query. Anchored to the
# start (not anywhere in the text) since "greeting" is about how the
# message *opens*, not whether it happens to contain one of these words.
GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|thx|good morning|good afternoon|good evening)\b",
    re.IGNORECASE,
)

# Digits, whitespace, and arithmetic operators/parens only -- "=" allowed
# since someone might type "2 + 2 =". Checked together with "contains a
# digit" so an empty/whitespace-only string can't match.
ARITHMETIC_RE = re.compile(r"^\s*[\d\s.()+\-*/=]+\s*$")

IDENTITY_RE = re.compile(r"\b(who are you|what are you|what can you do|what do you do)\b", re.IGNORECASE)

# --- keyword route patterns -------------------------------------------------

QUOTED_PHRASE_RE = re.compile(r'"[^"]+"|\'[^\']+\'')

# e.g. "JIRA-4532", "AB1234" -- ticket/code identifiers: a short run of
# uppercase letters, an optional separator, then 2+ digits.
IDENTIFIER_RE = re.compile(r"\b[A-Z]{1,6}[-_]?\d{2,}\b")

# e.g. "v2.1.0", "1.4" -- a dotted version number, "v" prefix optional.
VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")

# e.g. "chunk_pages", "build_vector_index".
SNAKE_CASE_RE = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")

# e.g. "getUserById" (lower-start) or "FastAPI"/"GetUserById" (upper-start,
# still has an internal case switch after a lowercase run).
CAMEL_CASE_RE = re.compile(r"\b(?:[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*)\b")

# e.g. "src/config.py", "data/raw/notes.txt", or a bare "filename.ext" token.
FILE_PATH_RE = re.compile(r"[./\\][\w./\\-]*\w|\b[\w-]+\.[a-zA-Z]{1,5}\b")

# e.g. "--max-findings", "--no-verify".
CLI_FLAG_RE = re.compile(r"--[a-zA-Z][\w-]*")

# --- single-fact lookup detection (used to pick the generator's context size) --

# "How many...", "What percentage...", "On how many...": the answer is one
# figure or name, so a few well-matched sentences beat five whole chunks.
FACTOID_START_RE = re.compile(
    r"^\s*(how (many|much)|on how many|by how (many|much)|over what|at what|when (was|did)"
    r"|what (percentage|percent|fraction|proportion|score|accuracy|correlation|spearman|pearson|number"
    r"|is the (number|size|score|accuracy|total|name)|were the (sizes|number))"
    r"|which (model|two|dataset|datasets))",
    re.IGNORECASE,
)

# Wording that asks for an explanation or comparison, which needs broader
# context than a few sentences, even if the question starts like a lookup.
EXPLANATORY_RE = re.compile(
    r"\b(why|explain|differ(ence|s|ent)?|compare|how (does|do|did|can|is|are)"
    r"|what (biases|limitations|shortcomings|drawbacks|contributions)"
    r"|limitations|contribution|conclude|motivate)\b",
    re.IGNORECASE,
)

QUESTION_WORDS = {"what", "why", "how", "when", "who", "which", "explain"}


def route(query: str) -> tuple[str, str]:
    """Classify `query` into a retrieval strategy.

    Returns (route, reason): route is one of "direct", "keyword", "hybrid";
    reason names which rule fired ("default" for the hybrid fallback).
    """
    if GREETING_RE.search(query):
        return "direct", "greeting_or_thanks"
    if ARITHMETIC_RE.match(query) and any(c.isdigit() for c in query):
        return "direct", "pure_arithmetic"
    if IDENTITY_RE.search(query):
        return "direct", "identity_question"

    if QUOTED_PHRASE_RE.search(query):
        return "keyword", "quoted_phrase"
    if IDENTIFIER_RE.search(query):
        return "keyword", "identifier_pattern"
    if VERSION_RE.search(query):
        return "keyword", "version_pattern"
    if SNAKE_CASE_RE.search(query):
        return "keyword", "snake_case_token"
    if CAMEL_CASE_RE.search(query):
        return "keyword", "camel_case_token"
    if FILE_PATH_RE.search(query):
        return "keyword", "file_path"
    if CLI_FLAG_RE.search(query):
        return "keyword", "cli_flag"

    words = query.strip().split()
    has_question_word = any(w.strip("?,.!").lower() in QUESTION_WORDS for w in words)
    if len(words) <= 3 and not has_question_word:
        return "keyword", "short_query_no_question_word"

    return "hybrid", "default"


def is_factoid_question(query: str) -> bool:
    """True for a single-fact lookup ("How many problems...?"); False for
    anything explanatory, multi-part or ambiguous.

    Deliberately conservative: it errs toward False, because a False just
    means the generator gets the full context (slower, never less accurate),
    while a wrong True would cut context an explanation needed. On the 76
    labelled in-scope questions it sent 40 of 49 lookup questions to the fast
    path and 0 of 27 conceptual ones.
    """
    return bool(FACTOID_START_RE.search(query)) and not EXPLANATORY_RE.search(query)
