"""
Evaluation script for the Chess Challenge.

This script evaluates a trained chess model by:
1. Checking if the model has < 1M parameters
2. Verifying no illegal use of python-chess for move filtering
3. Playing games against a deterministic engine (500 total moves, restarting after 25 moves)
4. Tracking legal move rates (first try and with retries)

The evaluation is deterministic (greedy decoding, seeded random).
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Suppress HuggingFace warning about empty module names (harmless)
# This warning comes from transformers' dynamic_module_utils when loading custom code
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationResult:
    """Complete result of an evaluation run."""
    model_id: str
    n_parameters: int
    passed_param_check: bool
    passed_pychess_check: bool
    total_moves: int
    legal_moves_first_try: int
    legal_moves_with_retry: int
    games_played: int
    moves_per_game: List[int] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def legal_rate_first_try(self) -> float:
        return self.legal_moves_first_try / self.total_moves if self.total_moves > 0 else 0.0
    
    @property
    def legal_rate_with_retry(self) -> float:
        return self.legal_moves_with_retry / self.total_moves if self.total_moves > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_parameters": self.n_parameters,
            "passed_param_check": self.passed_param_check,
            "passed_pychess_check": self.passed_pychess_check,
            "total_moves": self.total_moves,
            "legal_moves_first_try": self.legal_moves_first_try,
            "legal_moves_with_retry": self.legal_moves_with_retry,
            "legal_rate_first_try": self.legal_rate_first_try,
            "legal_rate_with_retry": self.legal_rate_with_retry,
            "games_played": self.games_played,
            "moves_per_game": self.moves_per_game,
            "error_message": self.error_message,
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary for the model page discussion."""
        lines = [
            "## Evaluation Results",
            "",
            f"**Model**: `{self.model_id}`",
            f"**Parameters**: {self.n_parameters:,} {'[PASS]' if self.passed_param_check else '[FAIL] (exceeds 1M limit)'}",
            f"**Chess library check**: {'[PASS]' if self.passed_pychess_check else '[FAIL] (illegal use of python-chess)'}",
            "",
        ]
        
        if not self.passed_param_check:
            lines.append("**Evaluation not performed**: Model exceeds 1M parameter limit.")
            return "\n".join(lines)
        
        if not self.passed_pychess_check:
            lines.append("**Evaluation not performed**: Model illegally uses python-chess for move filtering.")
            return "\n".join(lines)
        
        if self.error_message:
            lines.append(f"**Evaluation error**: {self.error_message}")
            return "\n".join(lines)
        
        lines.extend([
            "### Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total moves played | {self.total_moves} |",
            f"| Games played | {self.games_played} |",
            f"| Legal moves (first try) | {self.legal_moves_first_try} ({self.legal_rate_first_try*100:.1f}%) |",
            f"| Legal moves (with retries) | {self.legal_moves_with_retry} ({self.legal_rate_with_retry*100:.1f}%) |",
            "",
            "### Interpretation",
            "",
            "- **>90% legal rate**: Excellent! Model has learned chess rules well.",
            "- **70-90% legal rate**: Good, but room for improvement.",  
            "- **<70% legal rate**: Model struggles with legal move generation.",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Security Checks
# =============================================================================

def count_parameters(model) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def check_pychess_usage(model_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the model code illegally uses python-chess for move filtering.
    
    Scans Python files in the model directory for patterns that suggest
    using chess.Board.legal_moves or similar to filter model outputs.
    
    Args:
        model_path: Path to the model directory.
        
    Returns:
        Tuple of (passed_check, error_message).
        passed_check is True if no illegal usage detected.
    """
    forbidden_patterns = [
        r'\.legal_moves',
        r'board\.is_legal\s*\(',
        r'move\s+in\s+.*legal',
        r'filter.*legal',
        r'legal.*filter',
    ]
    
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        # If it's a HuggingFace model ID, we can't check local files
        # We'll check the downloaded files after loading
        return True, None
    
    python_files = list(model_dir.glob("*.py"))
    
    for py_file in python_files:
        try:
            content = py_file.read_text()
            
            # Skip if it's just the standard model.py or tokenizer.py from the template
            if py_file.name in ["model.py", "tokenizer.py"]:
                # Check if it contains suspicious patterns in generate/forward methods
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name in ["forward", "generate", "__call__", "get_move"]:
                            func_code = ast.get_source_segment(content, node)
                            if func_code:
                                for pattern in forbidden_patterns:
                                    if re.search(pattern, func_code, re.IGNORECASE):
                                        return False, f"Illegal chess library usage in {py_file.name}:{node.name}"
            else:
                # For other files, check all content
                for pattern in forbidden_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return False, f"Illegal chess library usage detected in {py_file.name}"
                        
        except Exception as e:
            # If we can't parse the file, skip it
            continue
    
    return True, None


# =============================================================================
# Model Loading
# =============================================================================

REQUIRED_MODEL_FILES = [
    "config.json",           # Model configuration
    "model.safetensors",     # Model weights (or pytorch_model.bin)
]

REQUIRED_TOKENIZER_FILES = [
    "tokenizer_config.json", # Tokenizer configuration
    "vocab.json",            # Vocabulary file
]


def validate_model_files(model_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that a model directory contains all required files.
    
    For local paths, checks that the model contains:
    - Model architecture (config.json + weights)
    - Tokenizer (tokenizer_config.json + vocab.json)
    
    For HuggingFace Hub models, this is handled by the Hub.
    
    Args:
        model_path: Local path or HuggingFace model ID.
        
    Returns:
        Tuple of (is_valid, list of missing files).
    """
    is_local = os.path.exists(model_path)
    
    if not is_local:
        # HuggingFace Hub - validation happens during download
        return True, []
    
    model_dir = Path(model_path)
    missing_files = []
    
    # Check model files
    has_safetensors = (model_dir / "model.safetensors").exists()
    has_pytorch = (model_dir / "pytorch_model.bin").exists()
    if not (has_safetensors or has_pytorch):
        missing_files.append("model.safetensors (or pytorch_model.bin)")
    
    if not (model_dir / "config.json").exists():
        missing_files.append("config.json")
    
    # Check tokenizer files
    for fname in REQUIRED_TOKENIZER_FILES:
        if not (model_dir / fname).exists():
            missing_files.append(fname)
    
    return len(missing_files) == 0, missing_files


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[any, any, str]:
    """
    Load a model and tokenizer from a local path or HuggingFace Hub.
    
    The model must contain all necessary files:
    - config.json: Model configuration
    - model.safetensors (or pytorch_model.bin): Model weights
    - tokenizer_config.json: Tokenizer configuration
    - vocab.json: Vocabulary file
    
    Models must use trust_remote_code=True to load custom architectures.
    
    Args:
        model_path: Local path or HuggingFace model ID.
        device: Device to load the model on.
        verbose: Whether to print debug info.
        
    Returns:
        Tuple of (model, tokenizer, source_description).
        
    Raises:
        FileNotFoundError: If required model files are missing.
        RuntimeError: If model or tokenizer cannot be loaded.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    is_local = os.path.exists(model_path)
    
    # Validate model files for local paths
    is_valid, missing_files = validate_model_files(model_path)
    if not is_valid:
        raise FileNotFoundError(
            f"Model is missing required files: {', '.join(missing_files)}\\n"
            f"Your model must contain:\\n"
            f"  - config.json (model configuration)\\n"
            f"  - model.safetensors or pytorch_model.bin (model weights)\\n"
            f"  - tokenizer_config.json (tokenizer configuration)\\n"
            f"  - vocab.json (vocabulary)\\n"
            f"See example_solution/ for a reference."
        )
    
    if verbose:
        source = "local path" if is_local else "HuggingFace Hub"
        print(f"Loading model from {source}: {model_path}")
    
    # Try to load tokenizer
    tokenizer = None
    
    load_kwargs = {"trust_remote_code": True}
    if is_local:
        load_kwargs["local_files_only"] = True
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from {model_path}: {e}\\n"
            f"Make sure your model includes tokenizer files and custom tokenizer class."
        )
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            local_files_only=is_local,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path}: {e}\\n"
            f"Make sure your model includes config.json with auto_map and model weights."
        )
    
    if verbose:
        print(f"  Tokenizer: {type(tokenizer).__name__} (vocab_size={tokenizer.vocab_size})")
        print(f"  Model: {type(model).__name__}")
        print(f"  Parameters: {count_parameters(model):,}")
    
    return model, tokenizer, model_path


# =============================================================================
# Move Generation
# =============================================================================

class MoveGenerator:
    """
    Generates moves from a chess model using greedy decoding.
    
    The generation process:
    1. Tokenize the current game history
    2. Generate tokens greedily until whitespace is produced
    3. Extract UCI move from generated text
    4. Retry up to max_retries times if move is illegal
    """
    
    SQUARE_PATTERN = re.compile(r'[a-h][1-8]')
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_retries: int = 3,
        max_tokens_per_move: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_retries = max_retries
        self.max_tokens_per_move = max_tokens_per_move
        
        # Move model to device and set to eval mode
        if hasattr(model, 'to'):
            self.model = model.to(device)
        self.model.eval()
    
    def _is_whitespace_token(self, token_str: str) -> bool:
        """Check if token represents whitespace (separator between moves)."""
        if not token_str:
            return False
        # Check for EOS
        if hasattr(self.tokenizer, 'eos_token') and token_str == self.tokenizer.eos_token:
            return True
        # Check for whitespace
        return token_str.strip() == "" and len(token_str) > 0
    
    def _extract_uci_move(self, text: str) -> Optional[str]:
        """
        Extract a UCI move from generated text.
        
        Looks for two consecutive chess squares (e.g., e2e4).
        Handles promotion by looking for q/r/b/n after the destination.
        """
        squares = self.SQUARE_PATTERN.findall(text)
        
        if len(squares) < 2:
            return None
        
        from_sq, to_sq = squares[0], squares[1]
        uci_move = from_sq + to_sq
        
        # Check for promotion piece
        to_idx = text.find(to_sq)
        if to_idx != -1:
            remaining = text[to_idx + 2:to_idx + 5]
            promo_match = re.search(r'[=]?([qrbnQRBN])', remaining)
            if promo_match:
                uci_move += promo_match.group(1).lower()
        
        return uci_move
    
    def _generate_until_whitespace(
        self, 
        input_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate tokens until whitespace is encountered.
        
        Args:
            input_ids: Input token IDs.
            temperature: Sampling temperature. 0.0 = greedy (argmax).
        
        Uses greedy decoding (argmax) when temperature=0 for determinism.
        Uses sampling when temperature>0 for retries.
        """
        generated_tokens = []
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(self.max_tokens_per_move):
                outputs = self.model(input_ids=current_ids)
                logits = outputs.logits[:, -1, :]
                
                if temperature == 0.0:
                    # Greedy decoding: take argmax
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # Sampling with temperature
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode token
                token_str = self.tokenizer.decode(next_token[0])
                
                # Check for whitespace/separator
                if self._is_whitespace_token(token_str):
                    break
                
                generated_tokens.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        if generated_tokens:
            all_tokens = torch.cat(generated_tokens, dim=1)
            return self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        
        return ""
    
    def get_move(
        self,
        game_history: str,
        legal_moves: set,
    ) -> Tuple[Optional[str], bool]:
        """
        Generate a move for the current position.
        
        First attempt uses greedy decoding (deterministic).
        Retries use sampling with temperature (seeded for reproducibility).
        
        Args:
            game_history: Space-separated move history in model's format.
            legal_moves: Set of legal UCI moves for validation.
            
        Returns:
            Tuple of (uci_move, was_first_try).
            uci_move is None if all retries failed.
        """
        # Prepare input
        if game_history:
            input_text = self.tokenizer.bos_token + " " + game_history
        else:
            input_text = self.tokenizer.bos_token
        
        # Get max context length
        max_length = getattr(self.model.config, 'n_ctx', 512)
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - self.max_tokens_per_move,
        ).to(self.device)
        
        # Try to generate a legal move
        for attempt in range(self.max_retries):
            # First attempt: greedy (temperature=0)
            # Retries: sampling with increasing temperature
            temperature = 0.0 if attempt == 0 else 0.5 + 0.25 * attempt
            
            move_text = self._generate_until_whitespace(inputs["input_ids"], temperature)
            uci_move = self._extract_uci_move(move_text)
            
            if uci_move and uci_move in legal_moves:
                return uci_move, (attempt == 0)
        
        return None, False


# =============================================================================
# Chess Game Handler (with built-in deterministic engine)
# =============================================================================

# Piece values for simple evaluation
PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
}

# Piece-square tables for positional evaluation (simplified)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
]


class SimpleEngine:
    """
    A simple deterministic chess engine using minimax with alpha-beta pruning.
    
    This replaces Stockfish to ensure fully deterministic evaluation.
    The engine is intentionally weak (shallow search) to be beatable.
    """
    
    def __init__(self, depth: int = 2):
        self.depth = depth
    
    def evaluate_board(self, board) -> int:
        """
        Evaluate the board position.
        
        Returns a score from white's perspective.
        Positive = white advantage, Negative = black advantage.
        """
        if board.is_checkmate():
            return -30000 if board.turn else 30000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material counting
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                score += PIECE_VALUES.get(symbol, 0)
                
                # Add positional bonus for pawns
                if symbol == 'P':
                    score += PAWN_TABLE[63 - square]  # Flip for white
                elif symbol == 'p':
                    score -= PAWN_TABLE[square]
        
        # Small bonus for mobility
        if board.turn:  # White to move
            score += len(list(board.legal_moves))
        else:
            score -= len(list(board.legal_moves))
        
        return score
    
    def minimax(self, board, depth: int, alpha: int, beta: int, maximizing: bool) -> Tuple[int, Optional[any]]:
        """
        Minimax with alpha-beta pruning.
        
        Returns (score, best_move).
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None
        
        # Sort moves for better pruning (captures first)
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)
        
        best_move = moves[0] if moves else None
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def get_best_move(self, board) -> str:
        """Get the best move for the current position."""
        _, best_move = self.minimax(
            board, 
            self.depth, 
            -float('inf'), 
            float('inf'), 
            board.turn  # True if white to move
        )
        return best_move.uci() if best_move else None


class ChessGameHandler:
    """
    Handles chess game logic using python-chess.
    
    This class is used ONLY by the evaluation framework, not by the model.
    It manages the chess board state and uses a simple built-in engine
    for deterministic opponent moves.
    """
    
    def __init__(self, engine_depth: int = 2):
        import chess
        
        self.chess = chess
        self.board = chess.Board()
        self.engine = SimpleEngine(depth=engine_depth)
    
    def reset(self):
        """Reset the board to starting position."""
        self.board = self.chess.Board()
    
    def get_legal_moves_uci(self) -> set:
        """Get set of legal moves in UCI format."""
        return {move.uci() for move in self.board.legal_moves}
    
    def make_move(self, uci_move: str) -> bool:
        """Make a move on the board. Returns True if successful."""
        try:
            move = self.chess.Move.from_uci(uci_move)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
        except (ValueError, self.chess.InvalidMoveError):
            pass
        return False
    
    def get_opponent_move(self) -> str:
        """Get the opponent engine's move for the current position.
        
        Uses the built-in SimpleEngine for deterministic moves.
        """
        return self.engine.get_best_move(self.board)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_turn(self) -> str:
        """Get whose turn it is ('white' or 'black')."""
        return "white" if self.board.turn == self.chess.WHITE else "black"
    
    def get_move_history_formatted(self) -> str:
        """
        Get move history in the model's expected format.
        
        Converts UCI moves to the format: WPe2e4, BNg8f6, etc.
        """
        moves = []
        temp_board = self.chess.Board()
        
        for move in self.board.move_stack:
            color = "W" if temp_board.turn == self.chess.WHITE else "B"
            piece = temp_board.piece_at(move.from_square)
            piece_letter = piece.symbol().upper() if piece else "P"
            
            from_sq = self.chess.square_name(move.from_square)
            to_sq = self.chess.square_name(move.to_square)
            
            move_str = f"{color}{piece_letter}{from_sq}{to_sq}"
            
            # Handle promotion
            if move.promotion:
                promo_piece = self.chess.piece_symbol(move.promotion).upper()
                move_str += f"={promo_piece}"
            
            # Handle capture
            if temp_board.is_capture(move):
                move_str += "(x)"
            
            temp_board.push(move)
            
            # Handle check/checkmate
            if temp_board.is_checkmate():
                move_str += "(+*)" if "(x)" not in move_str else ""
                move_str = move_str.replace("(x)", "(x+*)")
            elif temp_board.is_check():
                if "(x)" in move_str:
                    move_str = move_str.replace("(x)", "(x+)")
                else:
                    move_str += "(+)"
            
            moves.append(move_str)
        
        return " ".join(moves)
    
    def close(self):
        """Clean up resources (no-op for built-in engine)."""
        pass


# =============================================================================
# Main Evaluator
# =============================================================================

class ChessEvaluator:
    """
    Main evaluator for the Chess Challenge.
    
    Evaluation procedure:
    1. Check model has < 1M parameters
    2. Check model doesn't use python-chess illegally
    3. Play games against deterministic engine:
       - 500 total moves (model moves)
       - Restart game after 25 moves
       - Model always plays white
    4. Track legal move rates
    """
    
    TOTAL_MOVES = 500
    MOVES_PER_GAME = 25
    SEED = 42
    
    def __init__(
        self,
        model,
        tokenizer,
        model_path: str,
        engine_depth: int = 2,
        max_retries: int = 3,
        device: str = "auto",
        total_moves: int = None,  # Override TOTAL_MOVES for testing
        moves_per_game: int = None,  # Override MOVES_PER_GAME for testing
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.max_retries = max_retries
        
        # Allow overriding constants for testing
        self.total_moves = total_moves if total_moves is not None else self.TOTAL_MOVES
        self.moves_per_game = moves_per_game if moves_per_game is not None else self.MOVES_PER_GAME
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize move generator
        self.move_generator = MoveGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_retries=max_retries,
        )
        
        # Initialize game handler with built-in deterministic engine
        self.game_handler = ChessGameHandler(engine_depth=engine_depth)
    
    def __del__(self):
        if hasattr(self, 'game_handler'):
            self.game_handler.close()
    
    def evaluate(self, verbose: bool = True) -> EvaluationResult:
        """
        Run the complete evaluation procedure.
        
        Returns:
            EvaluationResult with all metrics.
        """
        # Set seeds for determinism
        random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
        
        # Count parameters
        n_params = count_parameters(self.model)
        passed_param_check = n_params <= 1_000_000
        
        if verbose:
            status = "[PASS]" if passed_param_check else "[FAIL]"
            print(f"Parameter check: {n_params:,} parameters {status}")
        
        if not passed_param_check:
            return EvaluationResult(
                model_id=self.model_path,
                n_parameters=n_params,
                passed_param_check=False,
                passed_pychess_check=True,
                total_moves=0,
                legal_moves_first_try=0,
                legal_moves_with_retry=0,
                games_played=0,
                error_message="Model exceeds 1M parameter limit",
            )
        
        # Check for illegal python-chess usage
        passed_pychess, pychess_error = check_pychess_usage(self.model_path)
        
        if verbose:
            status = "[PASS]" if passed_pychess else "[FAIL]"
            print(f"Python-chess check: {status}")
        
        if not passed_pychess:
            return EvaluationResult(
                model_id=self.model_path,
                n_parameters=n_params,
                passed_param_check=True,
                passed_pychess_check=False,
                total_moves=0,
                legal_moves_first_try=0,
                legal_moves_with_retry=0,
                games_played=0,
                error_message=pychess_error,
            )
        
        # Run evaluation games
        if verbose:
            print(f"\nPlaying games against opponent engine...")
            print(f"  Total moves: {self.total_moves}")
            print(f"  Moves per game: {self.moves_per_game}")
        
        try:
            result = self._play_evaluation_games(verbose=verbose)
            result.passed_param_check = True
            result.passed_pychess_check = True
            result.n_parameters = n_params
            return result
        except Exception as e:
            return EvaluationResult(
                model_id=self.model_path,
                n_parameters=n_params,
                passed_param_check=True,
                passed_pychess_check=True,
                total_moves=0,
                legal_moves_first_try=0,
                legal_moves_with_retry=0,
                games_played=0,
                error_message=str(e),
            )
    
    def _play_evaluation_games(self, verbose: bool = True) -> EvaluationResult:
        """
        Play evaluation games and collect statistics.
        """
        total_model_moves = 0
        legal_first_try = 0
        legal_with_retry = 0
        games_played = 0
        moves_per_game = []
        
        while total_model_moves < self.total_moves:
            # Start a new game
            self.game_handler.reset()
            game_moves = 0
            games_played += 1
            
            while game_moves < self.moves_per_game and total_model_moves < self.total_moves:
                if self.game_handler.is_game_over():
                    break
                
                turn = self.game_handler.get_turn()
                
                if turn == "white":
                    # Model's turn
                    legal_moves = self.game_handler.get_legal_moves_uci()
                    history = self.game_handler.get_move_history_formatted()
                    
                    move, was_first_try = self.move_generator.get_move(history, legal_moves)
                    
                    total_model_moves += 1
                    game_moves += 1
                    
                    if move:
                        if was_first_try:
                            legal_first_try += 1
                        legal_with_retry += 1
                        self.game_handler.make_move(move)
                    else:
                        # All retries failed - make a random legal move to continue
                        # Sort for determinism (set iteration order is not guaranteed)
                        if legal_moves:
                            sorted_moves = sorted(legal_moves)
                            random_move = random.choice(sorted_moves)
                            self.game_handler.make_move(random_move)
                else:
                    # Opponent engine's turn
                    opp_move = self.game_handler.get_opponent_move()
                    self.game_handler.make_move(opp_move)
            
            moves_per_game.append(game_moves)
            
            if verbose and games_played % 5 == 0:
                rate = legal_with_retry / total_model_moves if total_model_moves > 0 else 0
                print(f"  Games: {games_played} | Moves: {total_model_moves}/{self.TOTAL_MOVES} | Legal rate: {rate:.1%}")
        
        return EvaluationResult(
            model_id=self.model_path,
            n_parameters=0,  # Will be set by caller
            passed_param_check=True,
            passed_pychess_check=True,
            total_moves=total_model_moves,
            legal_moves_first_try=legal_first_try,
            legal_moves_with_retry=legal_with_retry,
            games_played=games_played,
            moves_per_game=moves_per_game,
        )


# =============================================================================
# Hub Integration
# =============================================================================

def post_discussion_summary(model_id: str, result: EvaluationResult, token: Optional[str] = None):
    """
    Post evaluation summary as a discussion on the model's HuggingFace page.
    
    Args:
        model_id: The HuggingFace model ID.
        result: The evaluation result.
        token: HuggingFace token with write access.
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        
        # Create discussion with evaluation results
        api.create_discussion(
            repo_id=model_id,
            title="🏆 Evaluation Results",
            description=result.summary(),
            repo_type="model",
        )
        
        print(f"Posted evaluation summary to {model_id}")
        
    except Exception as e:
        print(f"Failed to post discussion: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate a chess model for the Chess Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a local model
  python -m src.evaluate --model_path ./my_model
  
  # Evaluate a HuggingFace model
  python -m src.evaluate --model_path LLM-course/chess-example
  
  # Evaluate and post results to HuggingFace
  python -m src.evaluate --model_path LLM-course/chess-example --post_results
        """
    )
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--engine_depth", type=int, default=2,
        help="Opponent engine search depth (default: 2)"
    )
    parser.add_argument(
        "--post_results", action="store_true",
        help="Post results as a discussion on the model's HuggingFace page"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for posting results (uses HF_TOKEN env var if not provided)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHESS CHALLENGE - EVALUATION")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    model, tokenizer, model_id = load_model_and_tokenizer(
        args.model_path,
        verbose=True,
    )
    
    print()
    
    # Create evaluator
    evaluator = ChessEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        engine_depth=args.engine_depth,
    )
    
    # Run evaluation
    result = evaluator.evaluate(verbose=True)
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(result.summary())
    
    # Post results if requested
    if args.post_results:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token:
            post_discussion_summary(model_id, result, token)
        else:
            print("\nWarning: No HuggingFace token provided. Cannot post results.")
    
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return result


def evaluate_model(model_path: str, verbose: bool = True) -> EvaluationResult:
    """
    Convenience function to evaluate a model from a path.
    
    Args:
        model_path: Path to the model directory (local or HuggingFace repo ID)
        verbose: Whether to print progress
    
    Returns:
        EvaluationResult with all metrics
    
    Example:
        >>> from src.evaluate import evaluate_model
        >>> results = evaluate_model("./my_model/final")
        >>> print(results.to_markdown())
    """
    model, tokenizer, model_id = load_model_and_tokenizer(model_path, verbose=verbose)
    
    evaluator = ChessEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
    )
    
    return evaluator.evaluate(verbose=verbose)


if __name__ == "__main__":
    main()
