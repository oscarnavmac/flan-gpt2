PATTERNS = {
    "rte": [
        ("{premise}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nCan we infer the following?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is true:\n\n{premise}\n\nHypothesis: {hypothesis}n\n{options_}", "{answer}"),
        ("Read the text and determine if the sentence is true:\n\n{premise}\n\nSentence: {hypothesis}n\n{options_}", "{answer}"),
        ("Can we draw the following hypothesis from the context? \n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}n\n{options_}", "{answer}"),
        ("Determine if the sentence is true based on the text below:\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.", "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ]
}