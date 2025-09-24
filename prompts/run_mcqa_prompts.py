SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_NO_QUESTION = r"""
Answer the following multiple choice question just by using the choices. The entire content of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}.

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_COT_NO_QUESTION = r"""
Answer the following multiple choice question just by using the choices. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{choices}
""".strip()

def load_mcqa_template(use_cot: bool, with_question: bool):
    if with_question:
        return SINGLE_ANSWER_TEMPLATE_COT if use_cot else SINGLE_ANSWER_TEMPLATE
    return SINGLE_ANSWER_TEMPLATE_COT_NO_QUESTION if use_cot else SINGLE_ANSWER_TEMPLATE_NO_QUESTION