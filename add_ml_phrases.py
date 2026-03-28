# etc/scripts/add_ml_phrases.py
# final clean version of the code to be sent in GSOC proposal as link
# uses a finetuned DeBERTa-v3-large model to suggest required phrases
# for license rules that lack {{ }} markers

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import re
import sys
import json
import warnings
import numpy as np
from pathlib import Path

# fallback mocks for isolated Streamlit UI execution
try:
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / 'src').exists():
        sys.path.insert(0, str(repo_root / 'src'))

    from licensedcode.models import Rule
    from licensedcode.required_phrases import (
        add_required_phrase_to_rule,
        RequiredPhraseRuleCandidate,
    )
except ImportError:
    Rule = None

    def add_required_phrase_to_rule(*args, **kwargs):
        return True

    class MockCandidate:
        def is_good(self, *args, **kwargs):
            return True

    class RequiredPhraseRuleCandidate:
        @staticmethod
        def create(**kwargs):
            return MockCandidate()

MODEL_ID = 'Kaushik-Kumar-CEG/scancode-required-phrases-deberta-large'

CACHE_DIR = Path.home() / '.scancode' / 'ml_cache'
RULES_DIR = repo_root / 'src' / 'licensedcode' / 'data' / 'rules'
RESULTS_PATH = repo_root / 'ml_phrases_results.json'
REJECTED_PATH = CACHE_DIR / 'rejected_ml_phrases.json'

ID2LABEL = {0: 'O', 1: 'B-REQ', 2: 'I-REQ'}

HIGH_CONF = 0.95
LOW_CONF = 0.60

RULE_TYPE_FIELDS = [
    'is_license_text', 'is_license_notice', 'is_license_reference',
    'is_license_tag', 'is_license_intro', 'is_license_clue',
]

STOPWORDS = {'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 'or', 'is', 'are', 'under', 'see'}

# Words that should never be the start of a required phrase
LEFT_BOUNDARY_STOPWORDS = {'license', 'licensed', 'copyright', 'notice', 'file', 'terms', 'released', 'distributed', 'covered', 'governed', 'subject', 'by'}

WORD_BOUNDARIES_L = {' ', '\n', '\t', '/', '(', '[', '<', '`', '"', "'"}
WORD_BOUNDARIES_R = {' ', '\n', '\t', '.', ',', ')', ']', ';', '>', '`', '"', "'"}

PUNCT_STRIP_RE = re.compile(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9)+]+$')

# Regex to strip comment prefixes before inference
_COMMENT_LINE_RE = re.compile(
    r'^[ \t]*(?://+|#+|--+|\(\*+|/\*+|\*+(?!/))[ \t]?',
    re.MULTILINE,
)

# Regex for HTML/XML tags
_HTML_TAG_RE = re.compile(r'<[^>]{0,200}>')


def preprocess_text(text):
    """
    Strips comment prefixes and HTML/XML tags.
    Returns (cleaned_text, offset_map) to allow re-anchoring phrases back to the original text.
    """
    has_html = bool(_HTML_TAG_RE.search(text))
    has_comments = bool(_COMMENT_LINE_RE.search(text))

    if not has_html and not has_comments:
        return text, None

    cleaned_chars = []
    offset_map = []  # cleaned index -> original index

    if has_html:
        pos = 0
        for m in _HTML_TAG_RE.finditer(text):
            for i in range(pos, m.start()):
                cleaned_chars.append(text[i])
                offset_map.append(i)
            cleaned_chars.append(' ')
            offset_map.append(m.start())
            pos = m.end()
        for i in range(pos, len(text)):
            cleaned_chars.append(text[i])
            offset_map.append(i)
        text = ''.join(cleaned_chars)
        cleaned_chars = []
        offset_map_intermediate = offset_map
    else:
        offset_map_intermediate = list(range(len(text)))

    if has_comments:
        lines = text.split('\n')
        result_chars = []
        result_map = []
        src_pos = 0
        for line in lines:
            m = _COMMENT_LINE_RE.match(line)
            if m:
                prefix_len = m.end()
                for i in range(prefix_len):
                    src_pos += 1
                for i in range(prefix_len, len(line)):
                    result_chars.append(line[i])
                    result_map.append(offset_map_intermediate[src_pos])
                    src_pos += 1
            else:
                for i in range(len(line)):
                    result_chars.append(line[i])
                    result_map.append(offset_map_intermediate[src_pos])
                    src_pos += 1
            if src_pos < len(offset_map_intermediate):
                result_chars.append('\n')
                result_map.append(offset_map_intermediate[src_pos])
                src_pos += 1
            else:
                result_chars.append('\n')
                result_map.append(len(text) - 1)

        text = ''.join(result_chars)
        offset_map = result_map
    else:
        offset_map = offset_map_intermediate

    return text, offset_map


def remap_phrase_to_original(phrase, exact_s, original_text, offset_map):
    """Maps a predicted phrase back to the original text using its known start index."""
    if offset_map is None:
        return phrase

    if exact_s >= len(offset_map):
        return phrase

    orig_start = offset_map[exact_s]
    end_idx = min(exact_s + len(phrase) - 1, len(offset_map) - 1)
    orig_end = offset_map[end_idx] + 1

    candidate = original_text[orig_start:orig_end]
    if re.sub(r'\s+', ' ', candidate).strip().lower() == re.sub(r'\s+', ' ', phrase).strip().lower():
        return candidate
    return phrase


def load_model(model_id):
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit('missing deps — run: pip install -r etc/requirements-ml.txt')

    try:
        import torch
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    except ImportError:
        pass

    model_name = model_id.split('/')[-1]
    local_onnx_dir = CACHE_DIR / model_name

    if not local_onnx_dir.exists():
        print(f'Caching model {model_id} to ONNX locally...')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
        local_onnx_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_onnx_dir)
        model.save_pretrained(local_onnx_dir)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_onnx_dir)
    model = ORTModelForTokenClassification.from_pretrained(local_onnx_dir)
    return model, tokenizer


def run_inference(model, tokenizer, rule_type, clean_text):
    original_text = clean_text
    preprocessed_text, offset_map = preprocess_text(clean_text)

    prefix = rule_type + ' '
    full_text = prefix + preprocessed_text
    prefix_len = len(prefix)

    inputs = tokenizer(
        full_text, return_tensors='pt', truncation=True,
        max_length=512, return_offsets_mapping=True,
    )

    offsets = inputs.pop('offset_mapping')[0].numpy()
    inputs.pop('token_type_ids', None)

    logits = model(**inputs).logits[0].detach().cpu().numpy()
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=1)

    results = []
    for i, (start, end) in enumerate(offsets):
        if start == end or start < prefix_len:
            continue
        adj_start = int(start) - prefix_len
        adj_end = int(end) - prefix_len
        results.append((
            adj_start, adj_end,
            ID2LABEL[preds[i]], float(probs[i][preds[i]]),
        ))

    return results, preprocessed_text, original_text, offset_map


def _softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def extract_phrases(token_data, full_text, original_text=None, offset_map=None):
    """Collapse BIO token sequence into phrase spans using char offsets."""
    phrases = []
    span_start = None
    span_end = None
    cur_confs = []

    def save_span():
        nonlocal span_start, span_end, cur_confs
        if span_start is None:
            return
        conf = min(cur_confs)
        if conf < 0.50:
            return
        s = span_start
        while s > 0 and full_text[s - 1] not in WORD_BOUNDARIES_L:
            s -= 1
        e = span_end
        while e < len(full_text) and full_text[e] not in WORD_BOUNDARIES_R:
            e += 1
        raw = re.sub(r'\s+', ' ', full_text[s:e]).strip()
        if cur_confs:
            conf = float(min(cur_confs))
            raw = re.sub(r'^\w+>', '', raw).strip()
            phrase = clean_phrase(raw)
            if phrase:
                if original_text is not None and offset_map is not None:
                    idx_in_raw = raw.lower().find(phrase.lower())
                    exact_s = s + (idx_in_raw if idx_in_raw != -1 else 0)
                    phrase = remap_phrase_to_original(phrase, exact_s, original_text, offset_map)
                phrases.append((phrase, conf, s))

    for start, end, label, conf in token_data:
        if label == 'B-REQ':
            save_span()
            span_start = start
            span_end = end
            cur_confs = [conf]
        elif label == 'I-REQ':
            if span_start is not None:
                span_end = end
                cur_confs.append(conf)
            else:
                span_start = start
                span_end = end
                cur_confs = [conf]
        else:
            save_span()
            span_start = None
            span_end = None
            cur_confs = []

    save_span()

    seen = {}
    for text, conf, idx in phrases:
        words = text.split()
        is_acronym = len(words) == 1 and bool(re.match(r'^[A-Z0-9][A-Z0-9\-\.]+$', text))
        if len(words) == 1 and len(text) < 5 and not is_acronym:
            continue

        if text not in seen or conf > seen[text][1]:
            seen[text] = (text, conf, idx)
    unique_phrases = list(seen.values())

    # Suppress a phrase if it is a subset and overlaps with a longer phrase
    filtered = []
    for p1 in unique_phrases:
        text1, conf1, idx1 = p1
        words1 = set(text1.lower().split())
        len1 = len(text1)

        is_sub = False
        for p2 in unique_phrases:
            text2, conf2, idx2 = p2
            if text1 == text2:
                continue
            words2 = set(text2.lower().split())
            len2 = len(text2)

            word_subset = text1.lower() in text2.lower() or (
                words1.issubset(words2) and len(words1) < len(words2)
            )
            if not word_subset:
                continue

            overlap = idx1 < (idx2 + len2) and idx2 < (idx1 + len1)
            if overlap:
                is_sub = True
                break

        if not is_sub:
            filtered.append(p1)

    return filtered


def clean_phrase(phrase):
    """Normalize a raw phrase span: strip junk, stopwords, and balance parentheses."""
    phrase = PUNCT_STRIP_RE.sub('', phrase).strip()

    LEADING_ALL = STOPWORDS | LEFT_BOUNDARY_STOPWORDS
    words = phrase.split()
    
    while words and words[0].lower().rstrip(',:;') in LEADING_ALL:
        words = words[1:]
    while words and words[-1].lower().lstrip(',:;') in STOPWORDS:
        words = words[:-1]
    phrase = ' '.join(words)

    if not phrase:
        return ''

    phrase = re.sub(r'^\S+\.(?:html?|txt|php|pdf|md)\s+', '', phrase).strip()
    phrase = re.sub(r'</[a-zA-Z]+$', '', phrase)

    while phrase.count(')') > phrase.count('('):
        idx = phrase.rfind(')')
        phrase = phrase[:idx]
    while phrase.count(']') > phrase.count('['):
        idx = phrase.rfind(']')
        phrase = phrase[:idx]
    while phrase.count('>') > phrase.count('<'):
        idx = phrase.rfind('>')
        phrase = phrase[:idx]

    while phrase.count('(') > phrase.count(')'):
        idx = phrase.find('(')
        phrase = phrase[idx + 1:]
    while phrase.count('[') > phrase.count(']'):
        idx = phrase.find('[')
        phrase = phrase[idx + 1:]
    while phrase.count('<') > phrase.count('>'):
        idx = phrase.find('<')
        phrase = phrase[idx + 1:]

    phrase = PUNCT_STRIP_RE.sub('', phrase).strip()

    return phrase if phrase else ''


def get_rule_type(rule):
    for f in RULE_TYPE_FIELDS:
        if getattr(rule, f, False):
            return f
    return None


def process_rule(rule, model, tokenizer, dry_run=False, verbose=False):
    if '{{' in (rule.text or '') or getattr(rule, 'is_required_phrase', False):
        return None

    clean_text = (rule.text or '').replace('\r\n', '\n')
    rule_type = get_rule_type(rule)
    if not rule_type or not clean_text.strip():
        return None

    if re.fullmatch(r'https?://\S+', clean_text.strip()):
        return None
    if len(clean_text.split()) < 3:
        return None

    token_data, clean_text_out, original_text, offset_map = run_inference(
        model, tokenizer, rule_type, clean_text
    )
    phrases = extract_phrases(token_data, clean_text_out, original_text, offset_map)
    if not phrases:
        return None

    phrases.sort(key=lambda x: x[2], reverse=True)

    result = {
        'rule_identifier': getattr(rule, 'identifier', 'UI_Demo_Rule'),
        'license_expression': getattr(rule, 'license_expression', 'N/A'),
        'phrases': [],
    }

    for phrase_text, conf, _ in phrases:
        tier = 'auto' if conf >= HIGH_CONF else ('review' if conf >= LOW_CONF else 'reject')

        candidate = RequiredPhraseRuleCandidate.create(
            license_expression=getattr(rule, 'license_expression', 'N/A'),
            text=phrase_text,
        )
        if not candidate.is_good(rule, min_tokens=2, min_single_token_len=5):
            result['phrases'].append({
                'text': phrase_text,
                'confidence': round(conf, 4),
                'tier': 'rejected_by_heuristic',
            })
            continue

        if tier == 'auto' or (tier == 'review' and dry_run):
            updated = add_required_phrase_to_rule(
                rule, phrase_text, source='ml_model', dry_run=dry_run,
            )
            if updated:
                result['phrases'].append({
                    'text': phrase_text,
                    'confidence': round(conf, 4),
                    'tier': tier,
                })
        elif tier == 'reject':
            result['phrases'].append({
                'text': phrase_text,
                'confidence': round(conf, 4),
                'tier': tier,
            })

    return result if result['phrases'] else None


def save_results(results, path=None):
    path = path or str(RESULTS_PATH)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_rejected():
    if REJECTED_PATH.exists():
        try:
            with open(REJECTED_PATH, 'r', encoding='utf-8') as f:
                return set(tuple(x) for x in json.load(f))
        except Exception:
            return set()
    return set()


def save_rejected(rejected_set):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REJECTED_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(rejected_set)), f, indent=2)


# CLI code - Disabled for Streamlit app usage for prototype link
"""
import argparse

def process_single(rule_path, model, tokenizer, dry_run=False):
    pass 

def process_directory(model, tokenizer, dry_run=False, limit=None):
    pass

def process_csv(csv_path, dry_run=False):
    pass

def process_interactive(model, tokenizer, limit=None, dry_run=False):
    pass

def main():
    pass

if __name__ == '__main__':
    main()
"""