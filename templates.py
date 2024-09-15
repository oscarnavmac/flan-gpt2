PATTERNS = {
    "rte1": [
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
    ],

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
    ],

    "common_gen": [
        ("Concepts: {concepts}\n\nWrite a sentence that includes all these words.", "{target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these keywords?", "{target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these concepts?", "{target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts}", "{target}"),
        ("Write a sentence about the following things:\n\n{concepts}", "{target}"),
        ("Generate a sentence that includes all the following words: {concepts}", "{target}"),
        ("What are the keywords in the following sentence:\n\n{target}", "{concepts}"),
        ("What are the most important words in the following sentence:\n\n{target}", "{concepts}"),
        ("Identify the most salient words in this sentence:\n\n{target}", "{concepts}"),
        ("Generate a sentence, and then tell me the concepts included in that sentence.", "Sentence:\n{target}\n\nConcepts:\n{concepts_newline}"),
    ],

    "xsum": [
        ("Summarize:\n\n{text}", "{summary}"),
        ("Summarize this article:\n\n{text}", "{summary}"),
        ("Summarize this article in one sentence.\n\n{text}", "{summary}"),
        ("{text}\nWhat is a summary of this text?", "{summary}"),
        ("{text}\nWhat was that article about?", "{summary}"),
        ("{text}\n\nThis article was about:", "{summary}"),
        ("Article:{text}\n\nA summary of the above article is?", "{summary}"),
        ("Article:{text}\n\nSummarize the main points of that article.", "{summary}"),
        ("Write an article based on this summary:\n\n{summary}", "{text}"),
        ("Write an article based on this \"{summary}\"", "{text}"),
    ],

    "bool_q": [
        ("{text}\n\nCan we conclude that {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nIs it true that {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\n{question}?\n\n{options_}", "{answer}"),
        ("Text: {text}\n\nQuestion: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat's the best answer to this question: {question}?\n\n{options_}", "{answer}"),
        ("{text}\nBased on the above text, what's the best answer to this question: {question}?\n\n{options_}", "{answer}"),
        ("{text}\nAnswer this question, making sure that the answer is supposed by the text: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nIs the following statement correct based on the text\n\n{question}\n\n{options_}", "{answer}"),
        ("{title}\n\n{text}\n\nIs this statement correct \"{question}\"?\n\n{options_}", "{answer}"),
        ("Is it true that {question} based on the following text?\n\n{text}\n\n{options_}", "{answer}"),
    ],
}