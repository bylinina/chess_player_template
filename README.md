# Chess Transformer Player

Minimal template for the ML Chess assignment.

## Interface

Your class **must be named**:

TransformerPlayer

It must implement:

```python
get_move(self, fen: str) -> Optional[str]
```
Return: UCI move string (e2e4) OR None

## Example

```python
from player import TransformerPlayer

p = TransformerPlayer("MyBot")
move = p.get_move("starting FEN")
print(move)
```
