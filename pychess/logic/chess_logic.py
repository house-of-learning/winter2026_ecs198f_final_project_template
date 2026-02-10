from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CASTLE_KINGSIDE = "0-0"
CASTLE_QUEENSIDE = "0-0-0"


@dataclass(frozen=True)
class Move:
    fr: tuple[int, int]  # (row, col) origin
    to: tuple[int, int]  # (row, col) destination
    piece: str
    captured: str = ""
    is_capture: bool = False
    is_en_passant: bool = False
    is_castle: bool = False
    promotion: str = ""  # "Q" or "q" if promotion


@dataclass
class GameState:
    """Typed container for internal chess state, replacing a plain dict."""

    castle_w_k: bool = True
    castle_w_q: bool = True
    castle_b_k: bool = True
    castle_b_q: bool = True
    en_passant_target: Optional[tuple[int, int]] = None

    def copy(self) -> GameState:
        return GameState(
            castle_w_k=self.castle_w_k,
            castle_w_q=self.castle_w_q,
            castle_b_k=self.castle_b_k,
            castle_b_q=self.castle_b_q,
            en_passant_target=self.en_passant_target,
        )


# Coordinate helpers
# Board orientation: board[0][0] == a8, board[7][0] == a1
# Row 0 = rank 8, Row 7 = rank 1
# Col 0 = file a, Col 7 = file h


def alg_to_rc(alg: str) -> tuple:
    """Convert algebraic notation like 'e2' to (row, col)."""
    col = ord(alg[0]) - ord("a")
    row = 8 - int(alg[1])
    return row, col


def rc_to_alg(r: int, c: int) -> str:
    """Convert (row, col) to algebraic notation like 'e2'."""
    return chr(c + ord("a")) + str(8 - r)


# Piece / board helpers
def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def is_white(piece: str) -> bool:
    return piece != "" and piece.isupper()


def is_black(piece: str) -> bool:
    return piece != "" and piece.islower()


def color_of(piece: str) -> str:
    if piece == "":
        return ""
    return "w" if piece.isupper() else "b"


def same_color(a: str, b: str) -> bool:
    ca = color_of(a)
    cb = color_of(b)
    return ca != "" and ca == cb


def enemy_color(color: str) -> str:
    return "b" if color == "w" else "w"


def find_king(board, color: str) -> tuple:
    king = "K" if color == "w" else "k"
    for r in range(8):
        for c in range(8):
            if board[r][c] == king:
                return r, c
    return -1, -1  # should never happen in a valid game


# Attack detection
KNIGHT_OFFSETS = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]

BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS

KING_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def is_square_attacked(board, target_rc: tuple, by_color: str) -> bool:
    """Check if a square is attacked by any piece of `by_color`."""
    tr, tc = target_rc

    # 1) Pawn attacks
    if by_color == "w":
        # White pawns attack *upward* (lower row index)
        for dc in [-1, 1]:
            pr, pc = tr + 1, tc + dc  # white pawn would be below target
            if in_bounds(pr, pc) and board[pr][pc] == "P":
                return True
    else:
        # Black pawns attack *downward* (higher row index)
        for dc in [-1, 1]:
            pr, pc = tr - 1, tc + dc  # black pawn would be above target
            if in_bounds(pr, pc) and board[pr][pc] == "p":
                return True

    # 2) Knight attacks
    for dr, dc in KNIGHT_OFFSETS:
        nr, nc = tr + dr, tc + dc
        if in_bounds(nr, nc):
            p = board[nr][nc]
            if p.upper() == "N" and color_of(p) == by_color:
                return True

    # 3) Bishop / Queen diagonal rays
    for dr, dc in BISHOP_DIRS:
        r, c = tr + dr, tc + dc
        while in_bounds(r, c):
            p = board[r][c]
            if p != "":
                if color_of(p) == by_color and p.upper() in ("B", "Q"):
                    return True
                break  # blocked
            r += dr
            c += dc

    # 4) Rook / Queen straight rays
    for dr, dc in ROOK_DIRS:
        r, c = tr + dr, tc + dc
        while in_bounds(r, c):
            p = board[r][c]
            if p != "":
                if color_of(p) == by_color and p.upper() in ("R", "Q"):
                    return True
                break
            r += dr
            c += dc

    # 5) King adjacency
    for dr, dc in KING_OFFSETS:
        kr, kc = tr + dr, tc + dc
        if in_bounds(kr, kc):
            p = board[kr][kc]
            if p.upper() == "K" and color_of(p) == by_color:
                return True

    return False


def is_in_check(board, color: str) -> bool:
    """Is the king of `color` currently in check?"""
    king_pos = find_king(board, color)
    if king_pos == (-1, -1):
        return False
    return is_square_attacked(board, king_pos, enemy_color(color))


# Pseudo-legal move generation
def _gen_pawn_moves(board, r, c, state) -> list:
    """Generate pseudo-legal pawn moves from (r, c)."""
    moves = []
    piece = board[r][c]
    color = color_of(piece)
    direction = -1 if color == "w" else 1  # white moves up (row--)
    start_row = 6 if color == "w" else 1
    promo_row = 0 if color == "w" else 7
    promo_piece = "Q" if color == "w" else "q"

    # Single push
    nr = r + direction
    if in_bounds(nr, c) and board[nr][c] == "":
        if nr == promo_row:
            moves.append(
                Move(fr=(r, c), to=(nr, c), piece=piece, promotion=promo_piece)
            )
        else:
            moves.append(Move(fr=(r, c), to=(nr, c), piece=piece))

        # Double push (only if single push square is also empty)
        if r == start_row:
            nr2 = r + 2 * direction
            if in_bounds(nr2, c) and board[nr2][c] == "":
                moves.append(Move(fr=(r, c), to=(nr2, c), piece=piece))

    # Diagonal captures
    for dc in [-1, 1]:
        nr, nc = r + direction, c + dc
        if in_bounds(nr, nc):
            target = board[nr][nc]
            # Normal capture
            if target != "" and color_of(target) != color:
                if nr == promo_row:
                    moves.append(
                        Move(
                            fr=(r, c),
                            to=(nr, nc),
                            piece=piece,
                            captured=target,
                            is_capture=True,
                            promotion=promo_piece,
                        )
                    )
                else:
                    moves.append(
                        Move(
                            fr=(r, c),
                            to=(nr, nc),
                            piece=piece,
                            captured=target,
                            is_capture=True,
                        )
                    )
            # En passant
            if state.en_passant_target == (nr, nc):
                # The captured pawn is on the same row as the moving pawn
                captured_pawn = board[r][nc]
                moves.append(
                    Move(
                        fr=(r, c),
                        to=(nr, nc),
                        piece=piece,
                        captured=captured_pawn,
                        is_capture=True,
                        is_en_passant=True,
                    )
                )

    return moves


def _gen_offset_moves(
    board, r: int, c: int, offsets: list[tuple[int, int]]
) -> list[Move]:
    """Shared helper for knight and king non-castling moves."""
    moves: list[Move] = []
    piece = board[r][c]
    color = color_of(piece)

    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            target = board[nr][nc]
            if target == "":
                moves.append(Move(fr=(r, c), to=(nr, nc), piece=piece))
            elif color_of(target) != color:
                moves.append(
                    Move(
                        fr=(r, c),
                        to=(nr, nc),
                        piece=piece,
                        captured=target,
                        is_capture=True,
                    )
                )
    return moves


def _gen_knight_moves(board, r, c) -> list[Move]:
    return _gen_offset_moves(board, r, c, KNIGHT_OFFSETS)


def _gen_sliding_moves(board, r, c, directions) -> list:
    moves = []
    piece = board[r][c]
    color = color_of(piece)
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while in_bounds(nr, nc):
            target = board[nr][nc]
            if target == "":
                moves.append(Move(fr=(r, c), to=(nr, nc), piece=piece))
            elif color_of(target) != color:
                moves.append(
                    Move(
                        fr=(r, c),
                        to=(nr, nc),
                        piece=piece,
                        captured=target,
                        is_capture=True,
                    )
                )
                break
            else:
                break  # own piece blocks
            nr += dr
            nc += dc
    return moves


def _gen_king_moves(board, r, c, state: GameState) -> list[Move]:
    """Generate pseudo-legal king moves including castling candidates."""
    moves = _gen_offset_moves(board, r, c, KING_OFFSETS)
    piece = board[r][c]
    color = color_of(piece)

    # Castling candidates (will be validated later for check constraints)
    opp = enemy_color(color)
    if color == "w":
        # King must be on e1 == (7, 4)
        if r == 7 and c == 4:
            # King-side: e1 -> g1
            if state.castle_w_k:
                if board[7][5] == "" and board[7][6] == "":
                    if (
                        not is_square_attacked(board, (7, 4), opp)
                        and not is_square_attacked(board, (7, 5), opp)
                        and not is_square_attacked(board, (7, 6), opp)
                    ):
                        moves.append(
                            Move(fr=(7, 4), to=(7, 6), piece=piece, is_castle=True)
                        )
            # Queen-side: e1 -> c1
            if state.castle_w_q:
                if board[7][3] == "" and board[7][2] == "" and board[7][1] == "":
                    if (
                        not is_square_attacked(board, (7, 4), opp)
                        and not is_square_attacked(board, (7, 3), opp)
                        and not is_square_attacked(board, (7, 2), opp)
                    ):
                        moves.append(
                            Move(fr=(7, 4), to=(7, 2), piece=piece, is_castle=True)
                        )
    else:
        # King must be on e8 == (0, 4)
        if r == 0 and c == 4:
            # King-side: e8 -> g8
            if state.castle_b_k:
                if board[0][5] == "" and board[0][6] == "":
                    if (
                        not is_square_attacked(board, (0, 4), opp)
                        and not is_square_attacked(board, (0, 5), opp)
                        and not is_square_attacked(board, (0, 6), opp)
                    ):
                        moves.append(
                            Move(fr=(0, 4), to=(0, 6), piece=piece, is_castle=True)
                        )
            # Queen-side: e8 -> c8
            if state.castle_b_q:
                if board[0][3] == "" and board[0][2] == "" and board[0][1] == "":
                    if (
                        not is_square_attacked(board, (0, 4), opp)
                        and not is_square_attacked(board, (0, 3), opp)
                        and not is_square_attacked(board, (0, 2), opp)
                    ):
                        moves.append(
                            Move(fr=(0, 4), to=(0, 2), piece=piece, is_castle=True)
                        )

    return moves


def gen_pseudo_moves(board, from_rc, state) -> list:
    """Generate all pseudo-legal moves for the piece at from_rc."""
    r, c = from_rc
    piece = board[r][c]
    if piece == "":
        return []

    pt = piece.upper()
    if pt == "P":
        return _gen_pawn_moves(board, r, c, state)
    elif pt == "N":
        return _gen_knight_moves(board, r, c)
    elif pt == "B":
        return _gen_sliding_moves(board, r, c, BISHOP_DIRS)
    elif pt == "R":
        return _gen_sliding_moves(board, r, c, ROOK_DIRS)
    elif pt == "Q":
        return _gen_sliding_moves(board, r, c, QUEEN_DIRS)
    elif pt == "K":
        return _gen_king_moves(board, r, c, state)
    return []


# Apply move (copy approach — Option A)
def apply_move_copy(
    board: list[list[str]], state: GameState, move: Move
) -> tuple[list[list[str]], GameState]:
    """
    Apply a move on copies of board and state.
    Returns (new_board, new_state).
    """
    new_board = [row[:] for row in board]
    new_state = state.copy()

    fr_r, fr_c = move.fr
    to_r, to_c = move.to
    piece = move.piece

    # Move the piece
    new_board[to_r][to_c] = piece
    new_board[fr_r][fr_c] = ""

    if move.is_en_passant:
        # The captured pawn is on the same row as the moving pawn, same col as destination
        new_board[fr_r][to_c] = ""

    if move.is_castle:
        if to_c == 6:  # king-side
            rook = new_board[to_r][7]
            new_board[to_r][7] = ""
            new_board[to_r][5] = rook
        elif to_c == 2:  # queen-side
            rook = new_board[to_r][0]
            new_board[to_r][0] = ""
            new_board[to_r][3] = rook

    if move.promotion:
        new_board[to_r][to_c] = move.promotion

    # --- Update state ---

    # En passant target
    if piece.upper() == "P" and abs(to_r - fr_r) == 2:
        ep_r = (fr_r + to_r) // 2
        new_state.en_passant_target = (ep_r, fr_c)
    else:
        new_state.en_passant_target = None

    # Castling rights
    # If king moved, lose both rights for that side
    if piece == "K":
        new_state.castle_w_k = False
        new_state.castle_w_q = False
    elif piece == "k":
        new_state.castle_b_k = False
        new_state.castle_b_q = False

    # If a rook moved/captured from its starting square, lose that right
    if move.fr == (7, 0) or move.to == (7, 0):
        new_state.castle_w_q = False
    if move.fr == (7, 7) or move.to == (7, 7):
        new_state.castle_w_k = False
    if move.fr == (0, 0) or move.to == (0, 0):
        new_state.castle_b_q = False
    if move.fr == (0, 7) or move.to == (0, 7):
        new_state.castle_b_k = False

    return new_board, new_state


# Legal move filtering
def is_legal_move(board, state, move, color) -> bool:
    """
    A move is legal if, after applying it, the moving side's king is not in check.
    Castling checks are already handled in gen — so just verify no self-check.
    """
    new_board, _ = apply_move_copy(board, state, move)
    return not is_in_check(new_board, color)


def legal_moves_for_square(board, state, from_rc, color) -> list:
    """Get all legal moves for the piece at from_rc."""
    pseudo = gen_pseudo_moves(board, from_rc, state)
    return [m for m in pseudo if is_legal_move(board, state, m, color)]


def has_any_legal_move(board, state, color) -> bool:
    """Does `color` have at least one legal move?"""
    for r in range(8):
        for c in range(8):
            if board[r][c] != "" and color_of(board[r][c]) == color:
                moves = legal_moves_for_square(board, state, (r, c), color)
                if moves:
                    return True
    return False


# Notation builder


def move_to_extended_notation(move) -> str:
    """Build extended chess notation string for a move."""
    if move.is_castle:
        if move.to[1] == 6:  # king-side
            return CASTLE_KINGSIDE
        else:
            return CASTLE_QUEENSIDE

    parts = []

    # Piece prefix: pawn gets nothing, others get lowercase letter
    if move.piece.upper() != "P":
        parts.append(move.piece.lower())

    # Starting square
    parts.append(rc_to_alg(*move.fr))

    # Capture marker
    if move.is_capture:
        parts.append("x")

    # Destination square
    parts.append(rc_to_alg(*move.to))

    # Promotion
    if move.promotion:
        parts.append("=Q")

    return "".join(parts)


# ChessLogic class
class ChessLogic:
    def __init__(self):
        """
        Initalize the ChessLogic Object. External fields are board and result

        board -> Two Dimensional List of string Representing the Current State of the Board
            P, R, N, B, Q, K - White Pieces

            p, r, n, b, q, k - Black Pieces

            '' - Empty Square

        result -> The current result of the game
            w - White Win

            b - Black Win

            d - Draw

            '' - Game In Progress
        """
        self.board = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]
        self.result = ""

        # Internal state
        self.turn = "w"
        self.castle_w_k = True
        self.castle_w_q = True
        self.castle_b_k = True
        self.castle_b_q = True
        self.en_passant_target = None

    # helpers
    def _state(self) -> GameState:
        return GameState(
            castle_w_k=self.castle_w_k,
            castle_w_q=self.castle_w_q,
            castle_b_k=self.castle_b_k,
            castle_b_q=self.castle_b_q,
            en_passant_target=self.en_passant_target,
        )

    def _apply_state(self, gs: GameState) -> None:
        self.castle_w_k = gs.castle_w_k
        self.castle_w_q = gs.castle_w_q
        self.castle_b_k = gs.castle_b_k
        self.castle_b_q = gs.castle_b_q
        self.en_passant_target = gs.en_passant_target

    def play_move(self, move: str) -> str:
        """
        Function to make a move if it is a valid move.

        Args:
            move (str): The move which is proposed. Format: "{starting_square}{ending_square}"
                e.g. "e2e4"

        Returns:
            str: Extended Chess Notation for the move, if valid. Empty str if invalid.
        """
        # 1) Game already over?
        if self.result != "":
            return ""

        # 2) Validate move string
        if len(move) != 4:
            return ""
        start = move[:2]
        end = move[2:]
        if (
            start[0] not in "abcdefgh"
            or start[1] not in "12345678"
            or end[0] not in "abcdefgh"
            or end[1] not in "12345678"
        ):
            return ""

        # 3) Parse coordinates
        r1, c1 = alg_to_rc(start)
        r2, c2 = alg_to_rc(end)

        # 4) Check piece at start
        piece = self.board[r1][c1]
        if piece == "":
            return ""
        if color_of(piece) != self.turn:
            return ""

        # 5) Reject same-color destination
        dest = self.board[r2][c2]
        if dest != "" and same_color(piece, dest):
            return ""

        # 6) Generate legal moves for (r1, c1) and find matching move
        state = self._state()
        legal = legal_moves_for_square(self.board, state, (r1, c1), self.turn)
        matching_move = None
        for m in legal:
            if m.to == (r2, c2):
                matching_move = m
                break

        if matching_move is None:
            return ""

        # 7) Compute notation using pre-move state
        notation = move_to_extended_notation(matching_move)

        # 8) Apply move to actual board and update internal state
        new_board, new_state = apply_move_copy(self.board, state, matching_move)
        self.board = new_board
        self._apply_state(new_state)

        # 9) Update result (check for checkmate / stalemate for opponent)
        opp = enemy_color(self.turn)
        if not has_any_legal_move(self.board, self._state(), opp):
            if is_in_check(self.board, opp):
                self.result = self.turn  # checkmate — current player wins
            else:
                self.result = "d"  # stalemate

        # 10) Flip turn
        self.turn = opp

        # 11) Return notation
        return notation
