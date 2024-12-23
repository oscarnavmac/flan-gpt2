PATTERNS = {
    "anli": [
        ("{premise}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nCan we infer the following?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is true:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}", "{answer}"),
        ("Read the text and determine if the sentence is true:\n\n{premise}\n\nSentence: {hypothesis}\n\n{options_}", "{answer}"),
        ("Can we draw the following hypothesis from the context? \n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}", "{answer}"),
        ("Determine if the sentence is true based on the text below:\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.", "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    
    "squad": [
        ("Please answer a question about the following article about {title}:\n\n{context}\n\n{question}", "{answer}"),
        ("Read this and answer the question\n\n{context}\n\n{question}", "{answer}"),
        ("{context}\n{question}", "{answer}"),
        ("Answer a question about this article:\n{context}\n{question}", "{answer}"),
        ("Here is a question about this article: {context}\nWhat is the answer to this question: {question}", "{answer}"),
        ("Article: {context}\n\nQuestion: {question}", "{answer}"),
        ("Article: {context}\n\nNow answer this question: {question}", "{answer}"),
        ("{title}\n{context}\n\nQ: {question}", "{answer}"),
        ("Ask a question about {title}.", "{question}"),
        ("What is the title of this article:\n\n{context}", "{title}"),
    ],
    
    # NOT IMPLEMENTED BECAUSE THE DATASET IS ALREADY FORMATTED IN INSTRUCTIONS
    "python_code": [
        ("{instruction}", "{solution}") for _ in range(10)
    ],
    
    "cosmos_qa": [
        ("{context}\n\nQuestion: {question}\n{options_}", "{answer}"),
        ("{context}\n\n{question}\n{options_}", "{answer}"),
        ("{context}\n\nAnswer the following question: {question}\n{options_}", "{answer}"),
        ("{context}\n\nBased on the preceding passage, answer the following question {question}\n{options_}", "{answer}"),
        ("{context}\n\nGive answer the following question using evidence from the above passage: {question}\n{options_}", "{answer}"),
        ("Context:{context}\nQuestion {question}\nAnswer:\n{options_}", "{answer}"),
        ("Read the following article and answer the question.\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Answer the question about text:\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Write a question about the article\n\n{context}", "{question}"),
        ("{context}\n\nGenerate a question about the above context.", "{question}"),
    ],
    
    "coqa": [
        ("{story}\n\nAnswer the following questions:\n{numbered_questions}", "{numbered_answers}"),
        ("Read the text and answer the questions.\n\n{story}\n\n{numbered_questions}", "{numbered_answers}"),
        ("Answer the questions at the end based on the text.\n\n{story}\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{story}\n\nAnswer this series of questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{story}\n\nWhat are the answers to this following set of questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{story}\n\nNow, provide a numbered list of answers to these questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{story}\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{story}\n\n{numbered_questions}\n\nProvide a numbered list of answers.", "{numbered_answers}"),
        ("Make use of the article to answer the questions.\n\n{story}\n\n{numbered_questions}", "{numbered_answers}"),
        ("{story}\n\nBased on the article and the following list of answers, write a list of questions.\n\n{numbered_answers}", "{numbered_questions}"),
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
        ("Generate a sentence, and then tell me the concepts included in that sentence.", "Sentence:\n{target}\n\nConcepts:\n{concepts}"),
    ],

    "xsum": [
        ("Summarize:\n\n{document}", "{summary}"),
        ("Summarize this article:\n\n{document}", "{summary}"),
        ("Summarize this article in one sentence.\n\n{document}", "{summary}"),
        ("{document}\nWhat is a summary of this text?", "{summary}"),
        ("{document}\nWhat was that article about?", "{summary}"),
        ("{document}\n\nThis article was about:", "{summary}"),
        ("Article: {document}\n\nA summary of the above article is?", "{summary}"),
        ("Article: {document}\n\nSummarize the main points of that article.", "{summary}"),
        ("Write an article based on this summary:\n\n{summary}", "{document}"),
        ("Write an article based on this \"{summary}\"", "{document}"),
    ],

    "bool_q": [
        ("{passage}\n\nCan we conclude that {question}?\n\n{options_}", "{answer}"),
        ("{passage}\n\nIs it true that {question}?\n\n{options_}", "{answer}"),
        ("{passage}\n\n{question}?\n\n{options_}", "{answer}"),
        ("Text: {passage}\n\nQuestion: {question}?\n\n{options_}", "{answer}"),
        ("{passage}\n\nWhat's the best answer to this question: {question}?\n\n{options_}", "{answer}"),
        ("{passage}\nBased on the above text, what's the best answer to this question: {question}?\n\n{options_}", "{answer}"),
        ("{passage}\nAnswer this question, making sure that the answer is supposed by the text: {question}?\n\n{options_}", "{answer}"),
        ("{passage}\n\nIs the following statement correct based on the text\n\n{question}\n\n{options_}", "{answer}"),
        ("{passage}\n\nIs this statement correct \"{question}\"?\n\n{options_}", "{answer}"),
        ("Is it true that {question} based on the following text?\n\n{passage}\n\n{options_}", "{answer}"),
    ],
    
    "eng_spa": [
        ("How do you say \"{eng}\" in Spanish?", "{spa}"),
        ("{spa} How do you say this sentence in English?", "{eng}"),
        ("{eng} Say this using Spanish", "{spa}"),
        ("Translate from English to Spanish:\n\n{eng}", "{spa}"),
        ("Translate from Spanish to English:\n\n{spa}", "{eng}"),
        ("Translate \"{spa}\" from Spanish to English.", "{eng}"),
        ("Translate \"{eng}\" to Spanish.", "{spa}"),
        ("Translate the following.\n\nEnglish: {eng}\n\nSpanish:", "{spa}"),
        ("Write a sentence in English.", "{eng}"),
        ("Write a sentence in Spanish.", "{spa}"),
    ],
    
    "paws": [
        ("Here are two sentences:\n{sentence1}\n{sentence2}\nDo they have the same meaning?\n{options_}", "{answer}"),
        ("Here are two sentences:\n\n{sentence1}\n\n{sentence2}\nAre the two sentences saying the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nDo the above sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nPlease tell me if the sentences above mean the same.\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these sentences conveying the same meaning?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nIf the first sentence is true, is the second one also true?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these two sentences paraphrases of each other?\n{options_}", "{answer}"),
        ("Do the following two sentences have the same meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Do these two sentences mean the same thing?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Do these sentences have the same meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
    ],
    
    # NOT IMPLEMENTED BECAUSE THE DATASET IS ALREADY FORMATTED IN INSTRUCTIONS
    "quora": [
        ("{question}", "{answer}") for _ in range(10)
    ],
    
    # NOT IMPLEMENTED BECAUSE THE DATASET IS ALREADY FORMATTED IN INSTRUCTIONS
    "alpaca": [
        ("{instruction}", "{output}") for _ in range(10)
    ],
}