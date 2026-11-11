"""What the builder page shows for the demo graph, pinned in one place.

tests/test_web_parity.py checks each value against the page's own code
under Node.js. tests/test_docs.py checks that the learning pages quote
the values exactly as they stand here. With one list, a page cannot
drift away from the code without a test failing.
"""

WEB_DEMO = {
    "web-scan": "-6.664",     # best expected cost among the 200 scanned settings
    "web-refined": "-7.187",  # after refinement
    "web-gap": "0.813",       # refined value minus the certified optimum
    "web-success": "43.8%",   # chance that one measurement is optimal, at the refined angles
}
