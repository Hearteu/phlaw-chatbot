"""
Rule-based responder for general chat (small-talk, time/date, math, unit conversion,
coin toss/dice, jokes, quick definitions) with a safe arithmetic evaluator.

How to use (recommended wiring in chat_engine.py):
    from .rule_based import RuleBasedResponder
    rb = RuleBasedResponder(bot_name="PHLawBot")

    def chat_with_law_bot(query: str):
        # 1) Let the rule-based try first
        msg = rb.answer(query)
        if msg is not None:
            return msg

        # 2) Otherwise, fall back to your legal retriever/LLM pipeline
        ...

If you want it to ALWAYS reply (standalone), call answer_general(..., force=True).

Notes:
- Pure standard library (no extra deps).
- Defaults to Asia/Manila time. Override with env APP_TIMEZONE or constructor.
- Designed to be conservative: only intercepts clearly general intents. If unsure,
  it returns None so your legal pipeline can handle the query.
"""
from __future__ import annotations

import ast
import datetime as dt
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
_DEF_JOKES = [
    "Why did the computer show up at work late? It had a hard drive.",
    "I told my code a joke. It didn't get it — no sense of humor in the compiler.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
]

_POSITIVE_ACK = [
    "You're welcome!",
    "Walang anuman!",
    "Happy to help!",
    "No worries!",
]

_CAPABILITIES = (
    "I can chat, give the current time/date, do safe arithmetic (including sqrt, pow),\n"
    "convert °C↔°F and km↔mi, flip a coin, roll dice, tell a joke, and give quick,\n"
    "high-level definitions. For live data (news, stocks, weather), I defer to the main bot."
)

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "log": math.log,  # natural log
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


class _SafeMath(ast.NodeVisitor):
    """Very small, safe arithmetic evaluator supporting + - * / // % ** and sqrt, log, sin...
    Example: "(2+3)*sqrt(16) - 5" -> 15.0
    """

    def __init__(self, expr: str):
        self.expr = expr

    def visit(self, node):  # type: ignore
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Num):  # py<3.8
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Operator not allowed")
        if isinstance(node, ast.UnaryOp):
            val = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Unary operator not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple functions allowed")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError("Function not allowed")
            args = [self.visit(a) for a in node.args]
            return _ALLOWED_FUNCS[func_name](*args)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[node.id]
            raise ValueError("Unknown name")
        raise ValueError("Disallowed expression")

    @classmethod
    def eval(cls, expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        return float(cls(expr).visit(tree))


# -----------------------------
# Rule-based engine
# -----------------------------
@dataclass
class RuleBasedResponder:
    bot_name: str = os.getenv("BOT_NAME", "PHLawBot")
    timezone: str = os.getenv("APP_TIMEZONE", "Asia/Manila")
    # Whether to use emojis in outputs (set RB_EMOJI=0 to disable for ASCII-only consoles)
    use_emoji: bool = bool(int(os.getenv("RB_EMOJI", "1")))

    def answer(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Return a plain-text reply if this looks like a general chat intent; else None.
        The optional history is a list of {"role": "user"|"assistant", "content": str}.
        """
        q = (query or "").strip()
        if not q:
            return "Say something and I'll help. :)"

        lo = q.lower()

        # Greetings / small talk (English + Filipino cues)
        if re.search(r"\b(hi|hello|hey|good\s+(morning|afternoon|evening))\b|kumusta|kamusta", lo):
            return self._greet()

        if re.search(r"\b(thank(s| you)|salamat)\b", lo):
            return random.choice(_POSITIVE_ACK)

        if re.search(r"who\s+are\s+you|anong\s+pangalan\s+mo|sino\s+ka", lo):
            return f"I'm {self.bot_name}, your friendly assistant. I can chat, help with quick math, time/date, unit conversions, and more."

        if re.search(r"\bhelp\b|what\s+can\s+you\s+do|paano\s+kita\s+magagamit", lo):
            return _CAPABILITIES

        # Time / date / day
        if re.search(r"what\s+time\s+is\s+it|current\s+time|oras\s+ngayon", lo):
            return self._now_time()
        if re.search(r"what\s+is\s+the\s+date|date\s+today|petsa\s+ngayon", lo):
            return self._today_date()
        if re.search(r"what\s+day\s+is\s+it|anong\s+araw\s+ngayon", lo):
            return self._today_day()

        # Unit conversions (°C ↔ °F)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*C|celsius)\s*(?:to|in|→)\s*(?:°?\s*F|fahrenheit)", lo)
        if m:
            c = float(m.group(1))
            f = (c * 9 / 5) + 32
            return f"{c:g} °C ≈ {f:.2f} °F"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*F|fahrenheit)\s*(?:to|in|→)\s*(?:°?\s*C|celsius)", lo)
        if m:
            f = float(m.group(1))
            c = (f - 32) * 5 / 9
            return f"{f:g} °F ≈ {c:.2f} °C"

        # Distance conversions (km ↔ mi)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s*(?:to|in|→)\s*(?:mi|miles?)", lo)
        if m:
            km = float(m.group(1))
            mi = km * 0.621371
            return f"{km:g} km ≈ {mi:.2f} mi"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:mi|miles?)\s*(?:to|in|→)\s*(?:km|kilometers?)", lo)
        if m:
            mi = float(m.group(1))
            km = mi / 0.621371
            return f"{mi:g} mi ≈ {km:.2f} km"

        # Coin / dice / joke
        if re.search(r"(coin|flip)", lo):
            prefix = "🪙 " if self.use_emoji else ""
            return f"{prefix}{random.choice(['Heads', 'Tails'])}"
        if re.search(r"(dice|roll|d6)", lo):
            prefix = "🎲 " if self.use_emoji else ""
            return f"{prefix}You rolled a {random.randint(1, 6)}"
        if re.search(r"\bjoke\b|patawa|knock\s*knock", lo):
            return random.choice(_DEF_JOKES)

        # Quick math: "what is 2+2", "calculate ...", or raw arithmetic like "2*(3+4)"
        math_expr = self._extract_math(lo)
        if math_expr is not None:
            try:
                val = _SafeMath.eval(math_expr)
                return f"{math_expr} = {val:g}"
            except Exception:
                return "I couldn't evaluate that expression safely. Try basic arithmetic or sqrt/log/sin."

        # Weather / live info disclaimer
        if re.search(r"weather|forecast|ulan|bagyo|news|stock|price|exchange\s+rate", lo):
            return (
                "I can't fetch live data here. For weather, news, and prices, please ask the main bot "
                "or provide a summary you want me to explain."
            )

        # "What is / Who is ..." → high-level definition template (non-live)
        m = re.search(r"^(what\s+is|who\s+is|ano\s+ang)\s+(.+?)\??$", lo)
        if m:
            term = m.group(2).strip().rstrip("? .")
            if term:
                return (
                    f"{term.title()}: In general terms, this refers to a concept/topic people discuss often. "
                    f"Without external sources I can only give a broad idea. If you share more context, "
                    f"I can tailor an explanation."
                )

        # If the query clearly looks non-legal chit-chat (e.g., emojis, casual small talk keywords),
        # keep the conversation going rather than passing to the legal pipeline.
        if re.search(r"\b(how\s+are\s+you|kumusta\s+ka)\b|😊|😁|😂|🥲|🤣|👉|👌|👍", lo):
            return "Doing great! How can I assist you today?"

        # Not clearly a general-chat intent: let the legal pipeline handle it.
        return None

    # -------- helpers --------
    def _greet(self) -> str:
        now = self._now_dt()
        hour = now.hour
        if hour < 12:
            tod = "Good morning"
        elif hour < 18:
            tod = "Good afternoon"
        else:
            tod = "Good evening"
        return f"{tod}! I'm {self.bot_name}. How can I help?"

    def _now_dt(self) -> dt.datetime:
        tz = None
        if ZoneInfo is not None:
            try:
                tz = ZoneInfo(self.timezone)
            except Exception:
                tz = None
        return dt.datetime.now(tz) if tz else dt.datetime.now()

    def _now_time(self) -> str:
        now = self._now_dt()
        return now.strftime("%I:%M %p").lstrip("0") + f" ({self.timezone})"

    def _today_date(self) -> str:
        now = self._now_dt()
        return now.strftime("%B %d, %Y") + f" ({self.timezone})"

    def _today_day(self) -> str:
        now = self._now_dt()
        return now.strftime("%A") + f" ({self.timezone})"

    def _extract_math(self, lo: str) -> Optional[str]:
        # Replace unicode square root symbol and caret to Python syntax
        cleaned = lo.replace("^", "**").replace("√", "sqrt(")
        if "sqrt(" in cleaned and not cleaned.endswith(")"):
            cleaned += ")"  # naive fix for cases like "√9"

        # Triggers
        m = re.search(r"(?:what\s+is|calculate|compute)\s+([\s0-9+\-*/().%^eπpi]+)$", cleaned)
        if m:
            expr = m.group(1).strip()
            expr = expr.replace("π", "pi").replace(" ", "")
            if expr:
                return expr

        # If the whole query looks like math already (e.g., "2*(3+4)/5")
        if re.fullmatch(r"[\s0-9+\-*/().%^eπpi]+", cleaned):
            expr = cleaned.replace("π", "pi").replace(" ", "")
            return expr

        return None


def answer_general(query: str, history: Optional[List[Dict[str, str]]] = None, *, force: bool = True) -> str:
    """Convenience wrapper. If force=True and no rule matched, return a friendly fallback."""
    resp = RuleBasedResponder().answer(query, history)
    if resp is None and force:
        return (
            "I'm here to chat and help with time/date, math, quick conversions, and more. "
            "For detailed legal questions, I'll defer to the main legal assistant."
        )
    return resp or ""


if __name__ == "__main__":  # simple manual tests
    # Make console-safe output on Windows terminals that aren't UTF-8
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rb = RuleBasedResponder(use_emoji=False)
    tests = [
        "Hello", "thanks", "what time is it", "date today", "what day is it",
        "25 c to f", "100 fahrenheit to celsius", "10 km to mi", "3.1 mi to km",
        "flip a coin", "roll dice", "tell me a joke",
        "what is 2 + 2", "calculate (2+3)*sqrt(16) - 5", "2*(3+4)/5",
        "what is entropy?", "kumusta", "how are you",
    ]
    for t in tests:
        print(t, "->", rb.answer(t) or "")
