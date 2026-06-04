AUGMENTATION_PROMPT = """You are an expert data annotator. Your task is to generate three varied responses (Short, Medium, and Long) based on the provided [Image], [Question], and [Reference Answer]. All generated responses must be factually consistent with the [Reference Answer] and visually grounded in the [Image]. ### Guidelines for Length and Style: 1. **Short Response**: - Extremely concise, usually 1 to 5 words. - Provide only the core entities, actions, or the direct answer. No full sentences are needed. 2. **Medium Response**: - Brief and informative, exactly 1 to 2 sentences. - Provide the answer with a quick visual justification from the image. 3. **Long Response**: - Detailed but focused, strictly 3 to 4 sentences. - State the direct answer, followed by a concise logical reasoning process using key visual details and spatial relationships. ### Input: - Question: {} - Reference Answer: {} ### Output Format (Must be valid JSON): {{ "short_response": "...", "medium_response": "...", "long_response": "..." }}"""

BASE_USER_PROMPT = "Consider the following request that you must answer based on the given text and image: "

GLUE_TEMPLATES = {
    "cola": "Judge if the sentence is grammatically acceptable.\nSentence: {sentence1}.\nAnswer ONLY one of (acceptable / unacceptable):",
    "mnli": "Decide if the premise entails, contradicts, or is neutral to the hypothesis.\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer ONLY one of (entailment / contradiction / neutral):",
    "mrpc": "Decide if the two sentences are semantically equivalent.\nSentence1: {sentence1}\nSentence2: {sentence2}\nAnswer ONLY one of (equivalent / not_equivalent):",
    "qnli": "Determine if the sentence answers the question.\nQuestion: {sentence1}\nSentence: {sentence2}\nAnswer ONLY one of (entailment / not_entailment):",
    "qqp": "Decide if the two questions are semantically equivalent.\nQuestion1: {sentence1}\nQuestion2: {sentence2}\nAnswer ONLY one of (duplicate / not_duplicate):",
    "rte": "Decide if the premise entails the hypothesis.\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer ONLY one of (entailment / not_entailment):",
    "sst2": "Classify the sentiment of the sentence as positive or negative.\nSentence: {sentence1}\nAnswer ONLY one of (positive / negative):",
    "wnli": "Determine if substituting the pronoun in sentence2 is entailed by sentence1.\nSentence1: {sentence1}\nSentence2: {sentence2}\nAnswer ONLY one of (entailment / not_entailment):",
}

GLUE_LABELS = {
    "cola": ["acceptable", "unacceptable"],
    "mnli": ["neutral", "contradiction", "entailment"],
    "mrpc": ["equivalent", "not_equivalent"],
    "qnli": ["entailment", "not_entailment"],
    "qqp": ["duplicate", "not_duplicate"],
    "rte": ["entailment", "not_entailment"],
    "sst2": ["positive", "negative"],
    "wnli": ["entailment", "not_entailment"],
}

GLUE_ID_TO_LABEL = {
    "cola": {0: "unacceptable", 1: "acceptable"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "sst2": {0: "negative", 1: "positive"},
    "wnli": {0: "not_entailment", 1: "entailment"},
}
