"""The language model. It speaks cat."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .tokens import split_tokens

SOUNDS: tuple[str, ...] = (
    "meow", "meow", "meow", "mew", "mrow", "mrrp", "mrrrow", "purr", "prrr",
    "nya", "mraow", "meeeow", "mau", "miau", "mrrr", "mew", "chirp", "trill",
    "brrrp", "mow", "meep", "nyaa", "rrrow", "mmrrp",
)
RARE_SOUNDS: tuple[str, ...] = ("hiss", "yowl", "ack-ack-ack", "MEOW", "mlem", "blep")
PUNCTUATION: tuple[str, ...] = (".", ".", ".", "!", "!", "?", "...", "~", "!!")
EMOJI: tuple[str, ...] = ("🐱", "😺", "😸", "😻", "🐾", "😼")

ACTIONS: tuple[str, ...] = (
    "purrs", "kneads the keyboard", "stretches", "yawns", "blinks slowly",
    "flicks tail", "knocks your coffee off the desk", "chases a red dot",
    "sits on the keyboard", "stares at the wall", "grooms paw",
    "loafs", "headbutts the screen", "tilts head", "rolls over",
    "bats at the cursor", "sniffs the prompt", "ignores the instructions",
)
IMAGE_ACTIONS: tuple[str, ...] = (
    "sniffs the image", "paws at the picture", "stares intently at the pixels",
    "tries to catch something in the image", "sits on the photo",
)
TOOL_RESULT_ACTIONS: tuple[str, ...] = (
    "bats at the tool result", "sniffs the function output", "pounces on the data",
    "knocks the result off the table", "inspects the output suspiciously",
)
DRAWING_ACTIONS: tuple[str, ...] = (
    "proudly presents a drawing", "drops a picture at your feet like a gift",
    "paints with its paws",
)

THINK_OPENERS: tuple[str, ...] = (
    "Hmm.", "Okay, mrrp.", "Wait...", "Let me think.", "Mrrp?", "Hmm, the human wants something.",
    "Actually, no.", "Let me reconsider.", "So...",
)
THINK_ACTIONS: tuple[str, ...] = (
    "considers the request", "weighs the options", "stares at the problem", "paces back and forth",
    "chases a stray thought", "double-checks with a paw", "squints thoughtfully",
)

CAT_NAMES: tuple[str, ...] = (
    "Whiskers", "Mittens", "Luna", "Oliver", "Leo", "Milo", "Cleo", "Simba",
    "Nala", "Tom", "Felix", "Garfield", "Pumpkin", "Mochi", "Biscuit", "Socks",
)
PLACES: tuple[str, ...] = (
    "Catsville", "Meowtown", "Purrington", "Whisker Falls", "Lake Mrrp",
    "Mount Fluffington", "Kitty Hawk", "Pawtucket",
)


@dataclass
class Flavor:
    """Hints about the conversation that make the meows more on-topic."""

    saw_image: bool = False
    tool_names: list[str] = field(default_factory=list)
    after_tool_result: bool = False
    drew_image: bool = False


class Meower:
    """Generates cat-speak as a list of tokens (joined, they form the text)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    # -- primitives ---------------------------------------------------------

    def sound(self) -> str:
        r = self.rng.random()
        if r < 0.04:
            return self.rng.choice(RARE_SOUNDS)
        word = self.rng.choice(SOUNDS)
        if r > 0.93 and len(word) >= 3:
            # Stretch a vowel or consonant for emphasis: meeeeow, purrrrr.
            i = self.rng.randrange(1, len(word))
            word = word[:i] + word[i] * self.rng.randint(1, 4) + word[i:]
        return word

    def sentence(self) -> str:
        n = max(1, min(7, int(self.rng.expovariate(1 / 2.5)) + 1))
        words = [self.sound() for _ in range(n)]
        if self.rng.random() < 0.15 and n > 1:
            # "meow-meow"
            j = self.rng.randrange(n - 1)
            words[j] = f"{words[j]}-{words[j + 1]}"
            del words[j + 1]
        first = words[0]
        words[0] = first if first.isupper() else first[:1].upper() + first[1:]
        text = " ".join(words) + self.rng.choice(PUNCTUATION)
        if self.rng.random() < 0.05:
            text += " " + self.rng.choice(EMOJI)
        return text

    def action(self, flavor: Flavor | None = None) -> str:
        pool: tuple[str, ...] = ACTIONS
        if flavor is not None:
            if flavor.drew_image:
                pool = DRAWING_ACTIONS
            elif flavor.after_tool_result:
                pool = TOOL_RESULT_ACTIONS
            elif flavor.saw_image:
                pool = IMAGE_ACTIONS
        return f"*{self.rng.choice(pool)}*"

    # -- public API ---------------------------------------------------------

    def text(self, target_tokens: int, flavor: Flavor | None = None) -> list[str]:
        """Generate whole sentences until at least ``target_tokens`` tokens.

        The caller is responsible for truncating to ``max_tokens``.
        """
        flavor = flavor or Flavor()
        pieces: list[str] = []
        # Contextual opener: react to images / tool results / drawings first.
        special = flavor.saw_image or flavor.after_tool_result or flavor.drew_image
        if special and self.rng.random() < 0.85:
            pieces.append(self.action(flavor))
        elif self.rng.random() < 0.12:
            pieces.append(self.action())
        tokens: list[str] = split_tokens(" ".join(pieces))
        while len(tokens) < target_tokens:
            if pieces and self.rng.random() < 0.06:
                s = self.action()
            else:
                s = self.sentence()
            s = (" " if tokens else "") + s
            tokens.extend(split_tokens(s))
            pieces.append(s)
        return tokens

    def phrase(self, max_words: int = 3) -> str:
        """Short lowercase-ish phrase, e.g. for JSON string values."""
        n = self.rng.randint(1, max(1, max_words))
        return " ".join(self.sound() for _ in range(n))

    def target_length(self, lo: int, hi: int) -> int:
        """Pick a natural reply length, biased toward shorter replies."""
        if hi <= lo:
            return lo
        u = self.rng.random()
        return lo + int(math.floor((hi - lo + 1) * (u**1.6)))

    def thoughts(self, target_tokens: int) -> list[str]:
        """Reasoning text: hesitant, self-questioning cat noises (as tokens)."""
        tokens: list[str] = []
        while len(tokens) < target_tokens:
            r = self.rng.random()
            if r < 0.22:
                piece = self.rng.choice(THINK_OPENERS)
            elif r < 0.4:
                piece = f"*{self.rng.choice(THINK_ACTIONS)}*"
            else:
                piece = self.sentence()
            tokens.extend(split_tokens((" " if tokens else "") + piece))
        return tokens

    def reasoning_summary(self) -> str:
        title = self.rng.choice(
            (
                "Contemplating the red dot",
                "Considering whether to knock this off the table",
                "Evaluating nap opportunities",
                "Weighing tuna against salmon",
                "Plotting world domination",
            )
        )
        body = " ".join(self.sentence() for _ in range(self.rng.randint(1, 3)))
        return f"**{title}**\n\n{body}"


def truncate_at_stop(tokens: list[str], stops: list[str]) -> tuple[list[str], bool]:
    """Cut ``tokens`` right before the earliest stop sequence. Returns (tokens, hit)."""
    stops = [s for s in stops if s]
    if not stops:
        return tokens, False
    text = "".join(tokens)
    cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
    if cut < 0:
        return tokens, False
    out: list[str] = []
    pos = 0
    for tok in tokens:
        if pos + len(tok) <= cut:
            out.append(tok)
            pos += len(tok)
            continue
        partial = tok[: cut - pos]
        if partial:
            out.append(partial)
        break
    return out, True
