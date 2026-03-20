# etc/scripts/add_ml_phrases.py
#
# Uses a fine-tuned DeBERTa-v3-large model to suggest required phrases
# for license rules that don't have {{ }} markers yet.
#
# Model is pulled from HuggingFace Hub and cached locally on first run.
# Calls add_required_phrase_to_rule() internally so all the usual
# safeguards (is_good, has_ignorable_changes etc) still apply.
#
# Usage:
#   python etc/scripts/add_ml_phrases.py --rule src/licensedcode/data/rules/gpl-2.0_1.RULE
#   python etc/scripts/add_ml_phrases.py --all --limit 20
#   python etc/scripts/add_ml_phrases.py --all --dry-run
#   python etc/scripts/add_ml_phrases.py --interactive
#   python etc/scripts/add_ml_phrases.py --csv approved.csv
#
# After any non-dry run: scancode-reindex-licenses
# Dependencies: pip install -r etc/requirements-ml.txt

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import re
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

# --- STREAMLIT CLOUD COMPATIBILITY BLOCK ---
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
    # Graceful fallback for the isolated Streamlit UI demo.
    # Mocks the internal Scancode functions so the ML engine still runs on the web.
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
# -------------------------------------------

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

# FIX 2: added legal boundary words that should never be the START of a required phrase
LEFT_BOUNDARY_STOPWORDS = {'license', 'licensed', 'copyright', 'notice', 'file', 'terms'}

WORD_BOUNDARIES_L = {' ', '\n', '\t', '/', '(', '[', '<', '`', '"', "'"}
WORD_BOUNDARIES_R = {' ', '\n', '\t', '.', ',', ')', ']', ';', '>', '`', '"', "'"}

# strip leading/trailing non-alphanumeric, but preserve trailing ')' and '+' for version patterns
PUNCT_STRIP_RE = re.compile(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9)+]+$')

# FIX 4: comment-prefix patterns to strip before inference
_COMMENT_LINE_RE = re.compile(
    r'^[ \t]*(?://+|#+|--+|\(\*+|/\*+|\*+(?!/))[ \t]?',
    re.MULTILINE,
)

# FIX 5: HTML/XML tag pattern
_HTML_TAG_RE = re.compile(r'<[^>]{0,200}>')


def preprocess_text(text):
    """
    FIX 4+5: Strip comment prefixes and HTML/XML tags before inference.

    Returns (cleaned_text, offset_map) where offset_map[i] is the index in the
    original `text` that cleaned_text[i] came from — used to re-anchor phrases
    back to the original text for injection.
    """
    # --- HTML stripping (Fix 5) ---
    # Replace tags with a single space so character counts shift predictably.
    # We build a char-level map: cleaned_pos -> original_pos
    has_html = bool(_HTML_TAG_RE.search(text))
    has_comments = bool(_COMMENT_LINE_RE.search(text))

    if not has_html and not has_comments:
        # fast path — no preprocessing needed
        return text, None

    # Build offset map over the original string
    cleaned_chars = []
    offset_map = []  # cleaned index -> original index

    if has_html:
        pos = 0
        for m in _HTML_TAG_RE.finditer(text):
            # copy chars before this tag verbatim
            for i in range(pos, m.start()):
                cleaned_chars.append(text[i])
                offset_map.append(i)
            # replace tag with a single space (preserves word boundaries)
            cleaned_chars.append(' ')
            offset_map.append(m.start())
            pos = m.end()
        for i in range(pos, len(text)):
            cleaned_chars.append(text[i])
            offset_map.append(i)
        text = ''.join(cleaned_chars)
        # rebuild for comment pass
        cleaned_chars = []
        offset_map_stage2 = []
        # remap through comment stripping below using the already-mapped text
        # (simplification: do comment strip on html-cleaned text, compose maps)
        offset_map_intermediate = offset_map
    else:
        offset_map_intermediate = list(range(len(text)))

    if has_comments:
        # strip per-line comment prefixes
        lines = text.split('\n')
        result_chars = []
        result_map = []
        src_pos = 0
        for line in lines:
            m = _COMMENT_LINE_RE.match(line)
            if m:
                prefix_len = m.end()
                # skip the prefix chars
                for i in range(prefix_len):
                    src_pos += 1
                # copy rest of line
                for i in range(prefix_len, len(line)):
                    result_chars.append(line[i])
                    result_map.append(offset_map_intermediate[src_pos])
                    src_pos += 1
            else:
                for i in range(len(line)):
                    result_chars.append(line[i])
                    result_map.append(offset_map_intermediate[src_pos])
                    src_pos += 1
            # newline
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
    """
    FIX 4+5: Maps a phrase back to original text using its known exact start index.
    """
    if offset_map is None:
        return phrase

    # map start/end through offset_map using the exact known index
    orig_start = offset_map[exact_s]
    end_idx = min(exact_s + len(phrase) - 1, len(offset_map) - 1)
    orig_end = offset_map[end_idx] + 1

    candidate = original_text[orig_start:orig_end]
    # normalise whitespace for comparison
    if re.sub(r'\s+', ' ', candidate).strip().lower() == re.sub(r'\s+', ' ', phrase).strip().lower():
        return candidate
    return phrase


def load_model(model_id):
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit('missing deps — run: pip install -r etc/requirements-ml.txt')

    # suppress torch TracerWarnings from ONNX export (only relevant here)
    try:
        import torch
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    except ImportError:
        pass

    model_name = model_id.split('/')[-1]
    local_onnx_dir = CACHE_DIR / model_name

    if not local_onnx_dir.exists():
        print(f'first run: exporting {model_id} to ONNX (takes ~5-7 min, cached after)')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
        local_onnx_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_onnx_dir)
        model.save_pretrained(local_onnx_dir)
        print(f'cached to {local_onnx_dir}')
        return model, tokenizer

    print(f'loading cached ONNX from {local_onnx_dir}')
    tokenizer = AutoTokenizer.from_pretrained(local_onnx_dir)
    model = ORTModelForTokenClassification.from_pretrained(local_onnx_dir)
    return model, tokenizer


def run_inference(model, tokenizer, rule_type, clean_text):
    # FIX 4+5: preprocess before inference
    import inspect, add_ml_phrases
    st.write(inspect.getfile(add_ml_phrases))
    original_text = clean_text
    preprocessed_text, offset_map = preprocess_text(clean_text)

    prefix = rule_type + ' '
    full_text = prefix + preprocessed_text
    prefix_len = len(prefix)

    inputs = tokenizer(
        full_text, return_tensors='pt', truncation=True,
        max_length=512, return_offsets_mapping=True,
    )
    if inputs['input_ids'].shape[1] == 512:
        print('  WARNING: rule truncated at 512 tokens, tail ignored')

    offsets = inputs.pop('offset_mapping')[0].numpy()
    inputs.pop('token_type_ids', None)

    logits = model(**inputs).logits[0].detach().cpu().numpy()
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=1)

    results = []
    for i, (start, end) in enumerate(offsets):
        # skip special tokens (start==end) and tokens inside the prefix
        if start == end or start < prefix_len:
            continue
        adj_start = int(start) - prefix_len
        adj_end = int(end) - prefix_len
        results.append((
            adj_start, adj_end,
            ID2LABEL[preds[i]], float(probs[i][preds[i]]),
        ))

    # return preprocessed_text as the coord space for extract_phrases,
    # plus original_text and offset_map for remapping
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
        if span_start is None:
            return
        conf = min(cur_confs)
        if conf < 0.50:
            return
        # snap to word boundaries
        s = span_start
        while s > 0 and full_text[s - 1] not in WORD_BOUNDARIES_L:
            s -= 1
        e = span_end
        while e < len(full_text) and full_text[e] not in WORD_BOUNDARIES_R:
            e += 1
        raw = re.sub(r'\s+', ' ', full_text[s:e]).strip()
        if cur_confs:
            conf = float(min(cur_confs))
            # strip XML tag remnants (name>, type>, url>) at span start
            raw = re.sub(r'^\w+>', '', raw).strip()
            phrase = clean_phrase(raw)
            if phrase:
                # FIX 4+5: remap phrase back to original text coords using exact index
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
                # orphan I-REQ — treat as start of new span to maximise recall
                span_start = start
                span_end = end
                cur_confs = [conf]
        else:
            save_span()
            span_start = None
            span_end = None
            cur_confs = []

    save_span()

    # deduplicate — keep highest confidence when same phrase predicted twice
    seen = {}
    for text, conf, idx in phrases:
        words = text.split()
        # Drop single words under 5 chars UNLESS they are uppercase acronyms (e.g. MIT, GPL, ZPL)
        is_acronym = len(words) == 1 and bool(re.match(r'^[A-Z0-9][A-Z0-9\-\.]+$', text))
        if len(words) == 1 and len(text) < 5 and not is_acronym:
            continue

        if text not in seen or conf > seen[text][1]:
            seen[text] = (text, conf, idx)
    unique_phrases = list(seen.values())

    # FIX 1+3: subset filter — only suppress a phrase if it is BOTH a word-subset
    # AND overlaps in character position with the longer phrase.
    # Phrases at different positions that share words (e.g. "GNU General Public License"
    # appearing twice in a dual-license rule) must NOT be suppressed.
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

            # word-set subset check (same as before)
            word_subset = text1.lower() in text2.lower() or (
                words1.issubset(words2) and len(words1) < len(words2)
            )
            if not word_subset:
                continue

            # FIX 1+3: additionally require character-position overlap
            # p1 span: [idx1, idx1+len1), p2 span: [idx2, idx2+len2)
            # overlap exists if intervals intersect
            overlap = idx1 < (idx2 + len2) and idx2 < (idx1 + len1)
            if overlap:
                is_sub = True
                break

        if not is_sub:
            filtered.append(p1)

    return filtered


def clean_phrase(phrase):
    """Normalise a raw phrase span: strip junk, stopwords, balance parens."""
    phrase = PUNCT_STRIP_RE.sub('', phrase).strip()

    # strip leading/trailing stopwords
    words = phrase.split()
    while words and words[0].lower().rstrip(',:;') in STOPWORDS:
        words = words[1:]
    while words and words[-1].lower().lstrip(',:;') in STOPWORDS:
        words = words[:-1]
    phrase = ' '.join(words)

    # FIX 2: strip leading legal boundary words that are never the start of a
    # required phrase (e.g. "LICENSE LGPL-2.0-or-later" -> "LGPL-2.0-or-later")
    words = phrase.split()
    while words and words[0].lower().rstrip(',:;') in LEFT_BOUNDARY_STOPWORDS:
        words = words[1:]
    phrase = ' '.join(words)

    # strip XML remnants like </name or </comments
    phrase = re.sub(r'</[a-zA-Z]+$', '', phrase)

    # balance matching pairs — an unbalanced closing bracket usually means we captured
    # the start of a Markdown link: `License](http...`
    # an unbalanced opening bracket usually means we captured garbage before the license
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

    if not phrase:
        return ''
    return phrase


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

    # URL-only and very short rules have high FP rates — skip
    if re.fullmatch(r'https?://\S+', clean_text.strip()):
        return None
    if len(clean_text.split()) < 3:
        return None

    token_data, clean_text_out, original_text, offset_map = run_inference(
        model, tokenizer, rule_type, clean_text
    )
    phrases = extract_phrases(token_data, clean_text_out, original_text, offset_map)
    if not phrases:
        if verbose:
            print('  Rule contains no recognizable license identifiers.')
        return None

    # reverse sort by char position so earlier offsets aren't shifted by injection
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
            if verbose:
                print(f'  skipped (is_good): {phrase_text!r}')
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
                if verbose:
                    tag = '[dry-run] would inject' if dry_run else 'injected'
                    print(f'  {tag}: {phrase_text!r}  conf={conf:.2f}  tier={tier}')
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
    print(f'results -> {path}')


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


def process_single(rule_path, model, tokenizer, dry_run=False):
    rule_path = Path(rule_path).resolve()
    if not rule_path.exists():
        print(f'file not found: {rule_path}')
        return

    rule = Rule.from_file(str(rule_path))
    result = process_rule(rule, model, tokenizer, dry_run=dry_run, verbose=True)
    if result:
        print(json.dumps(result, indent=2))
        save_results([result])
    else:
        print('no phrases predicted or rule already has markers')


def process_directory(model, tokenizer, dry_run=False, limit=None):
    if dry_run:
        print('dry run — nothing will be written\n')

    print(f'scanning {RULES_DIR}...')
    unprotected = []
    for rule_file in sorted(RULES_DIR.glob('*.RULE')):
        try:
            if '{{' not in rule_file.read_text(encoding='utf-8', errors='ignore'):
                unprotected.append(rule_file)
        except OSError:
            continue

    print(f'{len(unprotected)} unprotected rule files found\n')

    processed = 0
    results = []
    rejected_set = load_rejected()

    for rule_file in unprotected:
        if limit and processed >= limit:
            break

        try:
            rule = Rule.from_file(str(rule_file))
        except Exception as e:
            print(f'  error loading {rule_file.name}: {e}')
            continue

        result = process_rule(rule, model, tokenizer, dry_run=dry_run)

        if result and result['phrases']:
            filtered_phrases = []
            for p in result['phrases']:
                if (rule_file.name, p['text']) not in rejected_set:
                    filtered_phrases.append(p)
            result['phrases'] = filtered_phrases
            if not result['phrases']:
                result = None

        processed += 1

        if result:
            results.append(result)
            phrases_str = ', '.join(
                f"{p['text']!r}({p['tier'][0].upper()},{p['confidence']:.2f})"
                for p in result['phrases']
            )
            print(f'  [{processed:4d}] {rule_file.name}: {phrases_str}')

    print(f'\ndone — {processed} rules processed, phrases found in {len(results)}')
    if not dry_run:
        print('next: run scancode-reindex-licenses')
    save_results(results)


def process_csv(csv_path, dry_run=False):
    import csv as csv_mod
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')

    print(f'loading approved phrases from {csv_path.name}...')
    if dry_run:
        print('dry run — nothing will be written\n')

    processed = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_mod.DictReader(f)
        if not reader.fieldnames or not {'rule', 'phrase'}.issubset(reader.fieldnames):
            sys.exit("CSV must have 'rule' and 'phrase' columns")

        for row in reader:
            identifier = row['rule']
            phrase_text = row['phrase']

            rule_file = RULES_DIR / f'{identifier}.RULE'
            if not rule_file.exists():
                rule_file = RULES_DIR / f'{identifier}.yml'
            if not rule_file.exists():
                print(f'  warning: rule file not found for {identifier}')
                continue

            try:
                rule = Rule.from_file(str(rule_file))
            except Exception as e:
                print(f'  error loading {identifier}: {e}')
                continue

            updated = add_required_phrase_to_rule(
                rule, phrase_text, source='human_reviewed', dry_run=dry_run,
            )
            if updated:
                tag = '[dry-run] would inject' if dry_run else 'injected'
                print(f'  {tag}: {phrase_text!r} -> {identifier}')
                processed += 1

    print(f'\ndone — {processed} rules updated from CSV')
    if not dry_run:
        print('next: run scancode-reindex-licenses')


def _case_insensitive_replace(text, old, new_template, count=1):
    """Replace `old` in `text` preserving original case, case-insensitive match."""
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text
    replacement = new_template.format(match.group(0))
    return pattern.sub(replacement, text, count=count)


def process_interactive(model, tokenizer, limit=None, dry_run=False):
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule as RichRule
    import difflib

    console = Console()

    console.print(f'[dim]scanning {RULES_DIR}...[/dim]')
    unprotected = []
    for rule_file in sorted(RULES_DIR.glob('*.RULE')):
        try:
            if '{{' not in rule_file.read_text(encoding='utf-8', errors='ignore'):
                unprotected.append(rule_file)
        except OSError:
            continue

    console.print(f'[bold]{len(unprotected)}[/bold] unprotected rules found')
    console.print('[dim]tip: run --all first, then --interactive for the rest[/dim]')
    console.print('[dim]Ctrl+C to stop — resumable (injected rules skipped next run)[/dim]\n')

    auto_approved = manually_approved = skipped = 0
    processed = 0
    rejected_set = load_rejected()

    try:
        for rule_file in unprotected:
            if limit and processed >= limit:
                break

            try:
                rule = Rule.from_file(str(rule_file))
            except Exception as e:
                console.print(f'[red]error loading {rule_file.name}: {e}[/red]')
                continue

            result = process_rule(rule, model, tokenizer, dry_run=True)

            if not result or not result['phrases']:
                processed += 1
                continue

            actionable = [p for p in result['phrases'] if p['tier'] in ('auto', 'review')]
            filtered_actionable = []
            for p in actionable:
                if (rule_file.name, p['text']) not in rejected_set:
                    filtered_actionable.append(p)

            if not filtered_actionable:
                processed += 1
                continue

            processed += 1

            for phrase_info in filtered_actionable:
                phrase_text = phrase_info['text']
                conf = phrase_info['confidence']
                tier = phrase_info['tier']

                # reload from disk — in-memory text is stale after injection
                try:
                    rule = Rule.from_file(str(rule_file))
                except Exception as e:
                    console.print(f'[red]error reloading {rule_file.name}: {e}[/red]')
                    continue

                tier_color = 'green' if tier == 'auto' else 'yellow'
                original = rule.text or ''

                # case-insensitive replacement for preview diff
                injected = _case_insensitive_replace(
                    original, phrase_text, '{{{{{0}}}}}',
                )
                orig_lines = [line + '\n' for line in original.splitlines()]
                new_lines = [line + '\n' for line in injected.splitlines()]
                diff = ''.join(difflib.unified_diff(
                    orig_lines, new_lines,
                    fromfile=f'a/{rule_file.name}',
                    tofile=f'b/{rule_file.name}',
                ))

                console.print(Panel(
                    f'[bold]{rule_file.name}[/bold]  [dim]{rule.license_expression}[/dim]\n'
                    f'predicted phrase: [bold cyan]{phrase_text}[/bold cyan]  '
                    f'[{tier_color}]{conf:.0%} · {tier}[/{tier_color}]',
                    expand=False,
                ))

                if diff.strip():
                    console.print(Syntax(diff.strip(), 'diff', theme='monokai'))
                else:
                    console.print(f'[dim]{injected[:200]}[/dim]')

                console.print(
                    '\n[bold][[green]y[/green]] approve  '
                    '[[red]n[/red]] skip  '
                    '[[yellow]e[/yellow]] edit phrase  '
                    '[[dim]q[/dim]] quit[/bold]'
                )

                choice = ''
                while choice not in ('y', 'n', 'e', 'q'):
                    try:
                        choice = console.input('> ').strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        raise KeyboardInterrupt

                if choice == 'q':
                    raise KeyboardInterrupt

                if choice == 'n':
                    rejected_set.add((rule_file.name, phrase_text))
                    save_rejected(rejected_set)
                    skipped += 1
                    console.print('[yellow]skipped and remembered (will not prompt next time)[/yellow]\n')
                    continue

                if choice == 'e':
                    console.print(RichRule('[dim]full rule text[/dim]'))
                    console.print(f'[dim]{original}[/dim]')
                    console.print(RichRule())
                    console.print(
                        f'[dim]current prediction: [cyan]{phrase_text}[/cyan]\n'
                        f'type the exact phrase as it appears above.\n'
                        f'press Enter to keep current.[/dim]'
                    )
                    try:
                        edited = console.input('> ').strip()
                        if edited:
                            phrase_text = edited
                    except (EOFError, KeyboardInterrupt):
                        raise KeyboardInterrupt

                if choice in ('y', 'e'):
                    updated = add_required_phrase_to_rule(
                        rule, phrase_text, source='ml_model', dry_run=dry_run,
                    )
                    if updated:
                        label = 'auto' if tier == 'auto' else 'manual'
                        console.print(f'[green]✓ injected ({label})[/green]\n')
                        if tier == 'auto':
                            auto_approved += 1
                        else:
                            manually_approved += 1
                    else:
                        console.print(
                            '[red]✗ phrase not found in rule text[/red]\n'
                            '[dim]tip: press e and type the exact text from the rule[/dim]\n'
                        )

    except KeyboardInterrupt:
        console.print('\n[yellow]stopped[/yellow]')

    table = Table(show_header=True, header_style='bold')
    table.add_column('Metric')
    table.add_column('Count', justify='right')
    total = auto_approved + manually_approved
    table.add_row('Rules scanned', str(processed))
    table.add_row('[bold]Total injected[/bold]', f'[bold]{total}[/bold]')
    table.add_row('[green]  Auto-approved[/green]', f'[green]{auto_approved}[/green]')
    table.add_row('[cyan]  Manually approved[/cyan]', f'[cyan]{manually_approved}[/cyan]')
    table.add_row('[dim]Skipped[/dim]', f'[dim]{skipped}[/dim]')
    console.print(table)
    console.print('\nnext step: [bold]scancode-reindex-licenses[/bold]')


def main():
    parser = argparse.ArgumentParser(
        description='predict required phrases for scancode license rules',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rule', help='path to a single .RULE file')
    group.add_argument('--all', action='store_true',
                       help='scan all unprotected rules')
    group.add_argument('--csv',
                       help='inject from a reviewed CSV (skips ML model)')
    group.add_argument('--interactive', action='store_true',
                       help='interactive review with rich diff')

    parser.add_argument('--limit', type=int, default=None,
                        help='max rules to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='preview without writing')

    args = parser.parse_args()

    if args.csv:
        process_csv(args.csv, dry_run=args.dry_run)
        return

    model, tokenizer = load_model(MODEL_ID)

    if args.interactive:
        process_interactive(model, tokenizer, limit=args.limit, dry_run=args.dry_run)
    elif args.rule:
        process_single(args.rule, model, tokenizer, dry_run=args.dry_run)
    else:
        process_directory(model, tokenizer, dry_run=args.dry_run, limit=args.limit)


if __name__ == '__main__':
    main()
