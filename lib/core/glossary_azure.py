"""
Glossary support for Azure Translator (plain machine translation, no prompt).

Before the request, glossary terms found in the source text are marked or replaced so Azure uses the glossary
wording. After the response the result is validated; if a term did not survive, the next attempt is tried.
Attempts are tried in order for each target language:
  - "dict":        Azure dynamic dictionary (<mstrans:dictionary translation="...">term</mstrans:dictionary>).
                   Official feature for language pairs that include English; the term stays in context, so the
                   grammar around it is the best of all strategies. Needs an explicit source language and HTML mode.
  - "direct":      term -> target-language term in place. Used for German targets of non-English pairs (Azure
                   inflects a real noun itself).
  - "placeholder": term -> QX1, QX2... (Azure keeps such tokens untouched), restored after translation.
"""

import html
import os
import re

from lib.core.glossary_manager import glossary_manager, _ASCII_WORD
from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Target languages of non-English pairs that try "direct" replacement before falling back to placeholders
DIRECT_TARGETS = {lang.strip() for lang in os.getenv("GLOSSARY_AZURE_DIRECT_TARGETS", "de").split(",") if lang.strip()}
USE_DYNAMIC_DICTIONARY = os.getenv("GLOSSARY_AZURE_USE_DICT", "true").lower() not in ("0", "false", "no")
MAX_REPLACEMENTS = 30
PLACEHOLDER_PREFIXES = ("QX", "ZZ", "XQ")

_CJK = r'[\u3000-\u9fff\uff00-\uffef]'
_CLOSING_PUNCT = r'[,.;:!?)，。；：！？、）]'
_HANGUL_PARTICLE = re.compile(r'^(으로|로|와|과|은|는|이|가|을|를)(?![가-힣])')
_FULLWIDTH_PUNCT = {'，': ', ', '。': '. ', '；': '; ', '：': ': ', '？': '? ', '！': '! ', '、': ', '}


def _term_regex(term):
    if _ASCII_WORD.match(term):
        return r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'
    return re.escape(term)


def find_spans(source_lang, target_lang, text):
    """Non-overlapping glossary matches in `text` as sorted (start, end, target_term); longest source term first."""
    terms = glossary_manager.terms_for(source_lang, target_lang)
    if not terms or not text:
        return []
    taken = [False] * len(text)
    spans = []
    for src_term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(_term_regex(src_term), text, flags=re.I):
            if not any(taken[m.start():m.end()]):
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                spans.append((m.start(), m.end(), terms[src_term]))
    return sorted(spans)[:MAX_REPLACEMENTS]


def _substitute(text, spans, replacements):
    """Replace each span with its replacement; a space is put between replacements that touch each other."""
    out, last = [], 0
    for i, ((start, end, _), replacement) in enumerate(zip(spans, replacements)):
        if i > 0 and start == spans[i - 1][1]:
            replacement = ' ' + replacement
        out += [text[last:start], replacement]
        last = end
    out.append(text[last:])
    return ''.join(out)


def _normalize(text, target_lang):
    """Full-width punctuation -> half-width for languages that do not use it, and tidy the spacing around it."""
    if target_lang in ('en', 'de', 'ko'):
        for full, half in _FULLWIDTH_PUNCT.items():
            text = text.replace(full, half)
        text = re.sub(r'\s+([,.;:!?)])', r'\1', text)
        text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _hangul_fix_particle(term, right):
    """Pick the particle (와/과, 은/는, 이/가, 을/를, 으로/로) that matches the last syllable of the inserted term."""
    last = term[-1]
    m = _HANGUL_PARTICLE.match(right)
    if not m or not '가' <= last <= '힣':
        return right
    final = (ord(last) - 0xAC00) % 28
    has_final = final != 0
    particle = m.group(1)
    if particle in ('으로', '로'):
        fixed = '로' if final in (0, 8) else '으로'
    else:
        pairs = {'와': ('과', '와'), '과': ('과', '와'), '은': ('은', '는'), '는': ('은', '는'),
                 '이': ('이', '가'), '가': ('이', '가'), '을': ('을', '를'), '를': ('을', '를')}
        fixed = pairs[particle][0 if has_final else 1]
    return fixed + right[len(particle):]


def _place_term(result, match, term, target_lang):
    """Put `term` where `match` was found and tidy the spacing Azure left around the placeholder."""
    left, right = result[:match.start()], result[match.end():]
    if target_lang in ('zh', 'ja'):
        if left.endswith(' ') and left.rstrip() and re.match(_CJK, left.rstrip()[-1]):
            left = left.rstrip()
        if right.startswith(' ') and right.lstrip() and re.match(_CJK, right.lstrip()[0]):
            right = right.lstrip()
    right = re.sub(rf'^ +(?={_CLOSING_PUNCT})', '', right)
    if target_lang == 'ko':
        right = _hangul_fix_particle(term, right)
    return left + term + right


def _pick_prefix(text):
    for prefix in PLACEHOLDER_PREFIXES:
        if not re.search(prefix + r'\s*[-_]?\s*\d', text, flags=re.I):
            return prefix
    return None


def _placeholder_attempt(text, spans, target_lang):
    prefix = _pick_prefix(text)
    if prefix is None:
        return None
    keys = [f"{prefix}{i}" for i in range(1, len(spans) + 1)]
    request_text = _substitute(text, spans, keys)

    def finish(result):
        for i, (_, _, term) in enumerate(spans, 1):
            pattern = re.compile(rf'{prefix}\s*[-_]?\s*{i}(?!\d)', re.I)
            matches = list(pattern.finditer(result))
            if len(matches) != 1:
                return None
            result = _place_term(result, matches[0], term, target_lang)
        return _normalize(result, target_lang)

    return request_text, False, finish


def _direct_attempt(text, spans, target_lang):
    terms = [term for _, _, term in spans]
    request_text = _substitute(text, spans, terms)

    def finish(result):
        return _normalize(result, target_lang) if all(term in result for term in terms) else None

    return request_text, False, finish


def _dictionary_attempt(text, spans, target_lang):
    """Azure dynamic dictionary markup. Needs textType=html, so the rest of the text is HTML-escaped."""
    out, last = [], 0
    for start, end, term in spans:
        out.append(html.escape(text[last:start], quote=False))
        out.append(f'<mstrans:dictionary translation="{html.escape(term, quote=True)}">'
                   f'{html.escape(text[start:end], quote=False)}</mstrans:dictionary>')
        last = end
    out.append(html.escape(text[last:], quote=False))
    terms = [term for _, _, term in spans]
    starts_with_term = spans[0][0] == 0

    def finish(result):
        result = html.unescape(re.sub(r'<[^>]+>', '', result))
        if 'mstrans' in result or not all(term.lower() in result.lower() for term in terms):
            return None
        result = _normalize(result, target_lang)
        if starts_with_term and target_lang == 'en' and result[:1].islower():
            result = result[0].upper() + result[1:]
        return result

    return ''.join(out), True, finish


def plan_attempts(source_lang, target_lang, text):
    """
    Attempts for translating `text` into one target language with the glossary applied.
    `source_lang` must be a concrete language (resolve "auto" first with glossary_manager.resolve_source).

    Returns a list of (strategy_name, request_text, is_html, finish) in the order they should be tried;
    `finish(azure_result)` returns the final translation, or None if a glossary term did not survive.
    An empty list means the glossary does not apply (no match) and the plain translation should be used.
    """
    if not source_lang or source_lang == 'auto' or source_lang == target_lang:
        return []
    spans = find_spans(source_lang, target_lang, text)
    if not spans:
        return []
    attempts = []
    involves_english = 'en' in (source_lang, target_lang)
    if involves_english:
        # HTML mode folds line breaks into spaces, so multi-line text uses the plain-text strategies only
        if USE_DYNAMIC_DICTIONARY and '\n' not in text:
            attempts.append(("dict", *_dictionary_attempt(text, spans, target_lang)))
    elif target_lang in DIRECT_TARGETS:
        attempts.append(("direct", *_direct_attempt(text, spans, target_lang)))
    placeholder = _placeholder_attempt(text, spans, target_lang)
    if placeholder:
        attempts.append(("placeholder", *placeholder))
    return attempts
