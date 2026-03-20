# review_ui/app.py

import traceback
import re
import sys
import html
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Scancode Required Phrases",
    page_icon="🔍",
    layout="centered",
)

# ── model pre-warm on boot ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights...")
def get_cached_model():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    
    MODEL_ID = "Kaushik-Kumar-CEG/scancode-required-phrases-deberta-large"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16   # ✅ FP16 (same as original)
    )
    
    return model, tokenizer


# ── updated examples (safe change) ────────────────────────────────────────────
EXAMPLES = {
    "LGPL-2 + GPL-2 (multi)": (
        "is_license_notice",
        "This library is free software; you can redistribute it and/or\n"
        "modify it under the terms of the GNU Library General Public\n"
        "License as published by the Free Software Foundation; either\n"
        "version 2 of the License, or (at your option) any later version.\n\n"
        "On Debian systems, the complete text of the GNU Library General Public\n"
        "License can be found in /usr/share/common-licenses/LGPL-2 file.\n\n"
        "However, many parts of this library are licensed differently:\n\n"
        "This program is free software; you can redistribute it and/or\n"
        "modify it under the terms of the GNU General Public License as\n"
        "published by the Free Software Foundation; either version 2 of the\n"
        "License, or (at your option) any later version.\n\n"
        "On Debian systems, the complete text of the GNU General Public\n"
        "License can be found in /usr/share/common-licenses/GPL-2 file."
    ),
    "OLDAP-2.5": (
        "is_license_reference",
        "OLDAP-2.5 https://spdx.org/licenses/OLDAP-2.5"
    ),
    "LGPL-2.0-or-later tag": (
        "is_license_tag",
        "SPDXLicenseIdentifier: LGPL-2.0-or-later"
    ),
    "LGPL source notice": (
        "is_license_notice",
        "All source code is licensed under the GNU Lesser General Public License"
    ),
    "Ambiguous": (
        "is_license_notice",
        "derived from ICU (http://www.icu-project.org)\n"
        "The full license is available here:\n"
        "  http://source.icu-project.org/repos/icu/icu/trunk/license.html"
    ),
}

TIER_COLOR = {"auto": "#22c55e", "review": "#f59e0b", "reject": "#ef4444"}
TIER_LABEL = {"auto": "Auto-Approvable", "review": "Requires Manual Review", "reject": "Low Confidence / Skip"}


def highlight_phrase(rule_text, phrase):
    safe_text   = html.escape(rule_text)
    safe_phrase = html.escape(phrase)
    escaped     = re.escape(safe_phrase).replace(r"\ ", r"\s+")
    
    highlighted = re.sub(
        f"({escaped})",
        r'<mark style="background:#fef3c7;padding:1px 4px;border-radius:3px;color:#92400e;font-weight:bold">\1</mark>',
        safe_text, flags=re.IGNORECASE, count=1,
    )
    return highlighted.replace("\n", "<br>")


def make_diff(original, phrase):
    import difflib
    injected   = original.replace(phrase, f"{{{{{phrase}}}}}", 1)
    orig_lines = original.splitlines(keepends=True)
    new_lines  = injected.splitlines(keepends=True)
    diff = list(difflib.unified_diff(orig_lines, new_lines,
                                     fromfile="a/rule.RULE", tofile="b/rule.RULE", lineterm=""))
    lines = []
    for line in diff:
        esc = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(f'<span style="color:#22c55e;font-weight:bold">{esc}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f'<span style="color:#ef4444">{esc}</span>')
        elif line.startswith("@@"):
            lines.append(f'<span style="color:#60a5fa">{esc}</span>')
        else:
            lines.append(f'<span style="color:#94a3b8">{esc}</span>')
    return "<br>".join(lines)


# ── UI (UNCHANGED from original except text polish) ───────────────────────────
st.title("Scancode Required Phrase Extractor")

rule_type = st.session_state.get("rule_type", "is_license_notice")

rule_text = st.text_area(
    "Rule text",
    height=160,
    key="rule_text",
)

predict_btn = st.button("Predict Required Phrase")

# ── inference (UNCHANGED) ─────────────────────────────────────────────────────
if predict_btn and rule_text.strip():
    with st.spinner("Running inference..."):
        try:
            from add_ml_phrases import run_inference, extract_phrases

            model, tokenizer = get_cached_model()

            if re.fullmatch(r'https?://\S+', rule_text.strip()):
                st.info('URL-only rules are skipped — model cannot extract phrases.')
                st.stop()
                
            token_data, clean_text = run_inference(model, tokenizer, rule_type, rule_text)
            phrases_raw = extract_phrases(token_data, clean_text)

            seen = {}
            for text, conf, idx in phrases_raw:
                if text not in seen or conf > seen[text][1]:
                    seen[text] = (text, conf, idx)

            phrases = list(seen.values())
            phrases.sort(key=lambda x: x[1], reverse=True)

            if not phrases:
                st.warning("No recognizable license identifiers found.")
            else:
                for phrase_text, conf, _ in phrases:
                    st.write(f"{phrase_text} ({conf:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc(), language="python")

elif predict_btn:
    st.warning("Please enter some rule text first.")
