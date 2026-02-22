import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformerPlayer:
    """
    Tiny LM baseline chess player.

    Contract:
        get_move(fen: str) -> Optional[str]
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        temperature: float = 0.7,
        max_new_tokens: int = 8,
    ):
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading (CRITICAL)
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading model {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Return ONE legal move in UCI format.

FEN: {fen}
Move:"""

    # -------------------------
    # Extract UCI move
    # -------------------------
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    # -------------------------
    # Fallback
    # -------------------------
    def _random_legal(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
        except Exception:
            return None

        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        self._load_model()

        prompt = self._build_prompt(fen)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt if echoed
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
                return move

        except Exception:
            pass

        # Safe fallback if model fails
        return self._random_legal(fen)
