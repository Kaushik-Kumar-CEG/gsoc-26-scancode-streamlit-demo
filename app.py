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
@st.cache_resource(show_spinner="Loading model...")
def get_cached_model():
    for _p in [
        Path(__file__).resolve().parent.parent / "etc" / "scripts",
        Path(__file__).resolve().parent / "etc" / "scripts",
        Path.cwd() / "etc" / "scripts",
        Path.cwd().parent / "etc" / "scripts",
    ]:
        if (_p / "add_ml_phrases.py").exists():
            sys.path.insert(0, str(_p))
            break

    from add_ml_phrases import load_model, MODEL_ID
    model, tokenizer = load_model(MODEL_ID)   # ✅ FIXED
    return model, tokenizer

_ = get_cached_model()  # force execution on boot


# ── example rules ─────────────────────────────────────────────────────────────
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
TIER_LABEL = {"auto": "High confidence", "review": "Moderate confidence", "reject": "Low confidence"}


def highlight_phrase(rule_text, phrase):
    safe_text   = html.escape(rule_text)
    safe_phrase = html.escape(phrase)
    escaped     = re.escape(safe_phrase).replace(r"\ ", r"\s+")
    return re.sub(
        f"({escaped})",
        r'<mark style="background:#fef3c7;padding:1px 4px;border-radius:3px">\1</mark>',
        safe_text, flags=re.IGNORECASE, count=1,
    )


def make_diff(original, phrase):
    import difflib
    injected   = original.replace(phrase, f"{{{{{phrase}}}}}", 1)
    orig_lines = original.splitlines(keepends=True)
    new_lines  = injected.splitlines(keepends=True)
    diff = list(difflib.unified_diff(orig_lines, new_lines,
                                     fromfile="before", tofile="after", lineterm=""))
    lines = []
    for line in diff:
        esc = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(f'<span style="color:#22c55e">{esc}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f'<span style="color:#ef4444">{esc}</span>')
        else:
            lines.append(f'<span style="color:#94a3b8">{esc}</span>')
    return "<br>".join(lines)   # ✅ FIXED


# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-bottom:2px'>"
    "<span style='font-size:0.8em;color:#64748b;letter-spacing:0.06em'>"
    "GSoC 2026 Proposal · AboutCode / scancode-toolkit"
    "</span></div>",
    unsafe_allow_html=True,
)
st.title("License Required Phrase Detector")
st.markdown(
    "<div style='margin-top:-10px;margin-bottom:4px'>"
    "<span style='font-size:0.95em;color:#94a3b8'>by "
    "<a href='https://github.com/Kaushik-Kumar-CEG' style='color:#94a3b8;text-decoration:none'>"
    "Kaushik Kumar</a></span>"
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Predicts which phrase in a `.RULE` file should be marked as a required phrase — "
    "preventing false positive license detections in scancode-toolkit. "
    "DeBERTa-v3-large · F1 0.7931 · 83.9% useful on unseen licenses."
)
st.divider()

# ── example loader ────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (label, (rtype, rtext)) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["rule_type"] = rtype
        st.session_state["rule_text"] = rtext

# ── input ─────────────────────────────────────────────────────────────────────
rule_type = st.session_state.get("rule_type", "is_license_notice")

rule_text = st.text_area(
    "Rule text",
    height=160,
    key="rule_text",
    placeholder="Paste the body of a .RULE file here (the text below the `---` separator)...",
    help="Do not include YAML frontmatter — just the license text body."
)

st.caption(
    "💡 `.RULE` file format: YAML frontmatter followed by `---` then the plain rule text."
)

predict_btn = st.button("Predict Required Phrase", type="primary", use_container_width=True)

# ── inference ─────────────────────────────────────────────────────────────────
if predict_btn and rule_text.strip():
    with st.spinner("Running inference..."):
        try:
            from add_ml_phrases import run_inference, extract_phrases
            model, tokenizer = get_cached_model()

            if re.fullmatch(r'https?://\S+', rule_text.strip()):
                st.info('URL-only rules are skipped — the model cannot extract a required phrase from a bare URL.')
                st.stop()

            token_data, clean_text = run_inference(model, tokenizer, rule_type, rule_text)
            phrases_raw = extract_phrases(token_data, clean_text)

            seen = {}
            for text, conf, idx in phrases_raw:
                if text not in seen or conf > seen[text][1]:
                    seen[text] = (text, conf, idx)

            phrases = list(seen.values())
            phrases.sort(key=lambda x: x[1], reverse=True)   # ✅ FIXED

            if not phrases:
                st.warning("No required phrases predicted for this rule.")
            else:
                st.divider()
                st.markdown(f"**{len(phrases)} candidate phrase(s) found:**")

                for phrase_text, conf, _ in phrases:
                    tier   = "auto" if conf >= 0.95 else ("review" if conf >= 0.60 else "reject")
                    color  = TIER_COLOR[tier]
                    label  = TIER_LABEL[tier]

                    st.markdown(
                        f'<div style="border-left:3px solid {color};padding:8px 14px;'
                        f'margin:8px 0;border-radius:0 6px 6px 0;background:#0f172a">'
                        f'<code style="font-size:1.05em;color:#e2e8f0">{html.escape(phrase_text)}</code>'
                        f'<br><span style="color:{color};font-size:0.85em">'
                        f'{conf:.0%} · {label}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<style>.stProgress > div > div > div > div {{background-color: {color};}}</style>',
                        unsafe_allow_html=True,
                    )
                    st.progress(float(conf))

                best = phrases[0][0]

                st.markdown("**Highlighted in rule text:**")
                st.markdown(
                    f'<div style="font-family:monospace;white-space:pre-wrap;font-size:0.9em;'
                    f'background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;'
                    f'border:1px solid #1e293b;line-height:1.6">'
                    f'{highlight_phrase(rule_text, best)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**Diff:**")
                st.markdown(
                    f'<div style="font-family:monospace;white-space:pre-wrap;font-size:0.85em;'
                    f'background:#000000;color:#e2e8f0;padding:14px;border-radius:6px;'
                    f'border:1px solid #334155;line-height:1.6">'
                    f'{make_diff(rule_text, best)}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc(), language="python")

elif predict_btn:
    st.warning("Please enter some rule text first.")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='color:#475569;font-size:0.78em'>"
    "Model: <a href='https://huggingface.co/Kaushik-Kumar-CEG/scancode-required-phrases-deberta-large' "
    "style='color:#475569'>scancode-required-phrases-deberta-large</a> · "
    "<a href='https://github.com/Kaushik-Kumar-CEG' style='color:#475569'>github.com/Kaushik-Kumar-CEG</a>"
    "</div>",
    unsafe_allow_html=True,
)
