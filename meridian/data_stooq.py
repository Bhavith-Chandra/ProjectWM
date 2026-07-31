"""Stooq independent-source fetcher — DISABLED.

Stooq's free daily-CSV endpoint (stooq.com/q/d/l/) is now gated behind a JavaScript
proof-of-work bot-detection challenge. We do not bypass bot-detection, so this source is
unavailable. Held-out out-of-sample validation therefore uses `meridian.heldout` (Yahoo on
never-trained-on assets) plus the Oxford-Man Realized Library where an accessible trusted
mirror exists. See meridian/heldout.py.
"""
