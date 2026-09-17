"""App-specific styling shared by app.py and app1.py.

`ui_theme` carries the design system itself (paper ground, Fraunces / Inter /
JetBrains Mono, the vermilion accent, every native Streamlit widget, and the
plotly layout via ``style_fig``). It is a verbatim copy of the portfolio-wide
theme file, so anything true of *this dashboard only* lives here instead — and
because app.py and app1.py are near-duplicates, both import it, so the two
cannot drift apart visually.

Usage, immediately after ``st.set_page_config(...)``::

    ui_theme.apply_theme()
    ui_chrome.apply_app_css()

and at every chart::

    ui_theme.style_fig(fig, title_font=ui_chrome.CHART_TITLE_FONT)
    st.plotly_chart(fig, use_container_width=True, theme=None)

Presentation only. Selectors hang off Streamlit's stable ``data-testid`` hooks,
never the generated emotion class names, and font-family is never set on
``span`` or ``*`` — that breaks the Material icon ligatures.
"""

from __future__ import annotations

import streamlit as st

import ui_theme

# Every chart here has a title; it reads as a heading, so it takes the display
# face. (Only pass this to titled figures — a title_font with no title makes
# Streamlit print "undefined".)
CHART_TITLE_FONT = dict(family=ui_theme.FONT_DISPLAY, color=ui_theme.INK, size=17)

# Map overlays carry meaning (risky vs safer route, serious vs other accident),
# so they take the portfolio's status colours instead of CSS red / green /
# orange. Tiles are left alone; folium.Icon markers only accept folium's own
# named colours, so those stay as they are.
MAP_BAD = "#b3261e"
MAP_WARN = "#a86a12"
MAP_OK = "#3f7a3a"

APP_CSS = """
<style>
/* folium maps: a paper card edge around the iframe (a ring, not a border, so
   the fixed map width does not change). st_folium is this app's only custom
   component, so the component testid is a safe fallback for the title hook. */
iframe[title^="streamlit_folium"], iframe[data-testid="stCustomComponentV1"] {
  border-radius: 16px; background: var(--card);
  box-shadow: 0 0 0 1px var(--line), var(--shadow-sm); }

/* route-scoring reasons (st.text) read as a mono readout */
[data-testid="stText"] {
  font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  font-size: .8rem; color: var(--ink-2); }

/* number inputs (severity prediction): one framed control. Streamlit draws
   the frame on the container and zeroes the inner input's border; the shared
   theme's input rule would re-border that inner input, so it is reset here. */
[data-testid="stNumberInputContainer"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stNumberInputContainer"]:focus-within {
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
.stApp [data-testid="stNumberInputContainer"] [data-baseweb="input"],
.stApp [data-testid="stNumberInputContainer"] [data-baseweb="input"]:focus-within {
  border: 0; border-radius: 0; box-shadow: none; background: transparent; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
  background: transparent; color: var(--ink-3); }
[data-testid="stNumberInputStepDown"]:hover:enabled, [data-testid="stNumberInputStepUp"]:hover:enabled,
[data-testid="stNumberInputStepDown"]:focus:enabled, [data-testid="stNumberInputStepUp"]:focus:enabled {
  background: var(--accent-tint); color: var(--accent); }

/* status alerts: the portfolio's ok / warn / bad / info, always on their tint.
   Streamlit paints the tint on the outer container; the kind is only named on
   an inner child, hence :has(). */
[data-testid="stAlertContainer"] { border: 1px solid transparent; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
  background-color: rgba(47, 95, 138, .09); border-color: rgba(47, 95, 138, .22); color: #2f5f8a; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  background-color: rgba(63, 122, 58, .10); border-color: rgba(63, 122, 58, .25); color: #3f7a3a; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
  background-color: rgba(168, 106, 18, .10); border-color: rgba(168, 106, 18, .25); color: #a86a12; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  background-color: rgba(179, 38, 30, .08); border-color: rgba(179, 38, 30, .22); color: #b3261e; }
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li { color: inherit; }

/* dropdown menus */
ul[role="listbox"] { border-radius: 12px !important;
  border: 1px solid var(--line) !important; background: var(--card) !important;
  box-shadow: var(--shadow-md) !important; padding: 5px !important; }
li[role="option"] { border-radius: 999px !important; font-size: .9rem !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background: var(--accent-tint) !important; color: var(--accent) !important; }
</style>
"""


def apply_app_css() -> None:
    """Inject this dashboard's app-specific rules on top of ui_theme."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
