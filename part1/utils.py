import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import hashlib
import re
from functools import lru_cache

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

# Nearby letter keys on a standard QWERTY keyboard.
_KEYBOARD_NEIGHBORS = {
    "q": "wa",
    "w": "qeas",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

# Match English words, treating contractions as a single token.
_WORD_PATTERN = re.compile(r"\b[A-Za-z]+(?:['’][A-Za-z]+)*\b")

# Preserve words that can be especially important to sentiment polarity.
_PROTECTED_WORDS = {
    "no", "not", "never", "neither", "nor",
    "none", "nobody", "nothing", "nowhere",
    "without", "hardly", "barely", "scarcely",
}

# Mutually exclusive operation probabilities.
_KEYBOARD_PROBABILITY = 0.15
_SWAP_PROBABILITY = 0.15
_SYNONYM_PROBABILITY = 0.10


@lru_cache(maxsize=20000)
def synonym_candidates(word):
    """Find single-word alternatives from the first WordNet sense."""
    synsets = wordnet.synsets(word)

    if not synsets:
        return ()

    sense = synsets[0]

    # Allow nouns, adjectives, adjective satellites, and adverbs.
    # Avoid verbs because replacements may break tense/agreement.
    if sense.pos() not in {"n", "a", "s", "r"}:
        return ()

    candidates = set()

    for lemma in sense.lemmas():
        candidate = lemma.name().lower()

        if (
            candidate != word
            and candidate.isascii()
            and candidate.isalpha()
            and candidate not in _PROTECTED_WORDS
        ):
            candidates.add(candidate)

    # Stable ordering ensures reproducible random selection.
    return tuple(sorted(candidates))


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE #####

    text = example["text"]

    # Stable per-review randomness, independent of global random state.
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big")
    rng = random.Random(seed)

    def keyboard_typo(word):
        """Replace one internal letter with a neighboring keyboard key."""
        chars = list(word)

        # Exclude the first and last characters.
        i = rng.randrange(1, len(chars) - 1)
        original = chars[i]

        replacement = rng.choice(
            _KEYBOARD_NEIGHBORS[original.lower()]
        )

        chars[i] = (
            replacement.upper()
            if original.isupper()
            else replacement
        )

        return "".join(chars)

    def adjacent_swap(word):
        """Swap two distinct adjacent internal letters."""
        chars = list(word)

        positions = [
            i
            for i in range(1, len(chars) - 2)
            if chars[i].lower() != chars[i + 1].lower()
        ]

        if not positions:
            return word

        i = rng.choice(positions)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return "".join(chars)

    def replace_synonym(word):
        """Replace a lowercase word with a WordNet alternative."""
        # Conservatively skip capitalized names and acronyms.
        # This also skips sentence-initial capitalized words.
        if not word.islower():
            return word

        candidates = synonym_candidates(word)

        if not candidates:
            return word

        return rng.choice(candidates)

    def transform_word(match):
        word = match.group(0)

        # Preserve short words, contractions, and negation-related words.
        if (
            len(word) < 4
            or "'" in word
            or "’" in word
            or word.lower() in _PROTECTED_WORDS
        ):
            return word

        choice = rng.random()

        if choice < _KEYBOARD_PROBABILITY:
            return keyboard_typo(word)

        elif choice < _KEYBOARD_PROBABILITY + _SWAP_PROBABILITY:
            return adjacent_swap(word)

        elif choice < (
            _KEYBOARD_PROBABILITY
            + _SWAP_PROBABILITY
            + _SYNONYM_PROBABILITY
        ):
            return replace_synonym(word)

        return word

    # Substitute words while preserving surrounding whitespace/punctuation.
    example["text"] = _WORD_PATTERN.sub(transform_word, text)

    ##### YOUR CODE ENDS HERE #######

    return example