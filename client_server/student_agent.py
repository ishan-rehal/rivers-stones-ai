"""
Student Agent baseline using alpha-beta search with inexpensive heuristics.
"""

import random
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================


def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows


def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def top_score_row() -> int:
    return 2


def bottom_score_row(rows: int) -> int:
    return rows - 3


def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    return (y == top_score_row()) and (x in score_cols)


def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    return (y == bottom_score_row(rows)) and (x in score_cols)


def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"


# ==================== STARTER HELPERS ====================


def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int,
                              score_cols: List[int]) -> List[Dict[str, Any]]:
    moves: List[Dict[str, Any]] = []
    piece = board[y][x]
    if piece is None or piece.owner != player:
        return moves
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if piece.side == "stone":
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                continue
            if board[ny][nx] is None:
                moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
            elif board[ny][nx].owner != player:
                px, py = nx + dx, ny + dy
                if (in_bounds(px, py, rows, cols)
                        and board[py][px] is None
                        and not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
        for ori in ("horizontal", "vertical"):
            moves.append({"action": "flip", "from": [x, y], "orientation": ori})
    else:
        moves.append({"action": "flip", "from": [x, y]})
        moves.append({"action": "rotate", "from": [x, y]})
    return moves


def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int,
                       score_cols: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player:
                out.extend(get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols))
    return out


# ==================== BASE AGENT ====================


class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        pass


# ==================== STUDENT AGENT ====================


class StudentAgent(BaseAgent):
    """Simple alpha-beta agent with quick heuristics and in-place simulation."""

    def __init__(self, player: str):
        super().__init__(player)
        # Evaluation weights - optimized values
        self.W_SCORE = 200.0  # Increased - scoring is most important
        self.W_REACH = 25.0   # Increased - being close to scoring is crucial
        self.W_MOBILITY = 0.8  # Increased - mobility is important
        self.W_POSITION = 1.2  # Increased - positioning matters more
        self.W_BLOCK = 1.0     # Increased - blocking opponent is valuable
        self.W_THREAT = 50.0   # New - detect immediate threats
        self.W_CONTROL = 2.0   # New - control of key areas

        # Search parameters
        self.SEARCH_DEPTH = 2  # Increased for better play
        self.TIME_PER_MOVE = 3.0
        self.BEAM_WIDTH = 30   # Slightly increased
        
        # Game phase detection
        self.opening_moves = 0
        self.endgame_threshold = 8  # pieces remaining per player
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 10000
        
        # Null move pruning parameters
        self.null_move_reduction = 2
        self.null_move_threshold = 3

    # -------- Engine helpers --------
    def _engine(self):
        from gameEngine import (
            compute_valid_targets,
            get_river_flow_destinations,
            is_opponent_score_cell,
            is_own_score_cell,
        )
        return {
            'compute_valid_targets': compute_valid_targets,
            'get_river_flow_destinations': get_river_flow_destinations,
            'is_opponent_score_cell': is_opponent_score_cell,
            'is_own_score_cell': is_own_score_cell,
        }

    # -------- Move generation --------
    def _enumerate_moves(self, board, player, rows, cols, score_cols) -> List[Dict[str, Any]]:
        eng = self._engine()
        moves: List[Dict[str, Any]] = []
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece or piece.owner != player:
                    continue
                info = eng['compute_valid_targets'](board, x, y, player, rows, cols, score_cols)
                for tx, ty in info.get('moves', set()):
                    if board[ty][tx] is None:
                        moves.append({"action": "move", "from": [x, y], "to": [tx, ty]})
                for (ofx, ofy), (pfx, pfy) in info.get('pushes', []):
                    moves.append({"action": "push", "from": [x, y], "to": [ofx, ofy], "pushed_to": [pfx, pfy]})
                if piece.side == "stone":
                    for ori in ("horizontal", "vertical"):
                        prev_side, prev_ori = piece.side, piece.orientation
                        piece.side = "river"; piece.orientation = ori
                        flow = eng['get_river_flow_destinations'](board, x, y, x, y, player, rows, cols, score_cols)
                        piece.side = prev_side; piece.orientation = prev_ori
                        if all(not eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols)
                               for dx, dy in flow):
                            moves.append({"action": "flip", "from": [x, y], "orientation": ori})
                else:
                    moves.append({"action": "flip", "from": [x, y]})
                    prev_ori = piece.orientation
                    piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
                    flow = eng['get_river_flow_destinations'](board, x, y, x, y, player, rows, cols, score_cols)
                    piece.orientation = prev_ori
                    if all(not eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols)
                           for dx, dy in flow):
                        moves.append({"action": "rotate", "from": [x, y]})
        return moves

    # -------- In-place apply / undo --------
    def _apply_inplace(self, board, move, player, rows, cols, score_cols):
        eng = self._engine()
        action = move.get("action")
        if action not in ("move", "push", "flip", "rotate"):
            return False, "unknown action"

        if action in ("move", "push"):
            fr = move.get("from"); to = move.get("to")
            if not fr or not to:
                return False, "bad format"
            fx, fy = int(fr[0]), int(fr[1])
            tx, ty = int(to[0]), int(to[1])
            if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
                return False, "oob"
            piece = board[fy][fx]
            if piece is None or piece.owner != player:
                return False, "invalid piece"
            info = eng['compute_valid_targets'](board, fx, fy, player, rows, cols, score_cols)
            moves = info.get('moves', set())
            pushes = set(info.get('pushes', []))
            if action == "push" or move.get("pushed_to") is not None:
                pushed_to = move.get("pushed_to")
                if not pushed_to:
                    return False, "pushed_to required"
                px, py = int(pushed_to[0]), int(pushed_to[1])
                if ((tx, ty), (px, py)) not in pushes:
                    return False, "push pair invalid"
                pushed_piece = board[ty][tx]
                mover_piece = board[fy][fx]
                board[py][px] = pushed_piece
                board[ty][tx] = mover_piece
                board[fy][fx] = None
                flipped = False
                prev_side = mover_piece.side
                prev_ori = mover_piece.orientation
                if mover_piece.side == "river":
                    flipped = True
                    mover_piece.side = "stone"
                    mover_piece.orientation = None
                undo = {
                    'kind': 'push',
                    'fx': fx, 'fy': fy, 'tx': tx, 'ty': ty,
                    'px': px, 'py': py,
                    'pushed_piece': pushed_piece,
                    'mover_piece': mover_piece,
                    'flipped': flipped,
                    'prev_side': prev_side,
                    'prev_ori': prev_ori,
                }
                return True, undo
            if (tx, ty) not in moves or board[ty][tx] is not None:
                return False, "illegal move"
            mover_piece = board[fy][fx]
            board[ty][tx] = mover_piece
            board[fy][fx] = None
            undo = {
                'kind': 'move',
                'fx': fx, 'fy': fy, 'tx': tx, 'ty': ty,
                'mover_piece': mover_piece,
            }
            return True, undo

        fr = move.get("from")
        if not fr:
            return False, "bad format"
        fx, fy = int(fr[0]), int(fr[1])
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"

        if action == "flip":
            if piece.side == "stone":
                ori = move.get("orientation")
                if ori not in ("horizontal", "vertical"):
                    return False, "orientation required"
                prev_side, prev_ori = piece.side, piece.orientation
                piece.side = "river"
                piece.orientation = ori
                flow = eng['get_river_flow_destinations'](board, fx, fy, fx, fy, player, rows, cols, score_cols)
                for dx, dy in flow:
                    if eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols):
                        piece.side = prev_side
                        piece.orientation = prev_ori
                        return False, "unsafe flip"
                undo = {"kind": "flip", "x": fx, "y": fy, "prev_side": prev_side, "prev_ori": prev_ori, "piece": piece}
                return True, undo
            prev_side, prev_ori = piece.side, piece.orientation
            piece.side = "stone"
            piece.orientation = None
            undo = {"kind": "flip", "x": fx, "y": fy, "prev_side": prev_side, "prev_ori": prev_ori, "piece": piece}
            return True, undo

        if action == "rotate":
            if piece.side != "river":
                return False, "rotate only on river"
            prev_ori = piece.orientation
            piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
            flow = eng['get_river_flow_destinations'](board, fx, fy, fx, fy, player, rows, cols, score_cols)
            for dx, dy in flow:
                if eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols):
                    piece.orientation = prev_ori
                    return False, "unsafe rotate"
            undo = {"kind": "rotate", "x": fx, "y": fy, "prev_ori": prev_ori}
            return True, undo

        return False, "unknown"

    def _undo_inplace(self, board, undo):
        kind = undo['kind']
        if kind == 'move':
            fx, fy, tx, ty = undo['fx'], undo['fy'], undo['tx'], undo['ty']
            mover = undo['mover_piece']
            board[fy][fx] = mover
            board[ty][tx] = None
        elif kind == 'push':
            fx, fy = undo['fx'], undo['fy']
            tx, ty = undo['tx'], undo['ty']
            px, py = undo['px'], undo['py']
            mover = undo['mover_piece']
            pushed = undo['pushed_piece']
            if undo.get('flipped'):
                mover.side = undo['prev_side']
                mover.orientation = undo['prev_ori']
            board[fy][fx] = mover
            board[ty][tx] = pushed
            board[py][px] = None
        elif kind == 'flip':
            x, y = undo['x'], undo['y']
            piece = undo.get('piece') or board[y][x]
            if board[y][x] is None:
                board[y][x] = piece
            piece.side = undo['prev_side']
            piece.orientation = undo['prev_ori']
        elif kind == 'rotate':
            x, y = undo['x'], undo['y']
            piece = board[y][x]
            piece.orientation = undo['prev_ori']

    # -------- Evaluation --------
    def _evaluate(self, board, rows, cols, score_cols) -> float:
        eng = self._engine()
        
        # Count scored pieces (O(156) - fast)
        n_self = 0
        n_opp = 0
        my_pieces = []
        opp_pieces = []
        
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p:
                    continue
                if p.owner == self.player:
                    my_pieces.append((x, y, p))
                else:
                    opp_pieces.append((x, y, p))
                
                if p.side == 'stone' and eng['is_own_score_cell'](x, y, p.owner, rows, cols, score_cols):
                    if p.owner == self.player:
                        n_self += 1
                    else:
                        n_opp += 1
        
        # Immediate win/loss detection
        if n_self >= 4:
            return 10000.0
        if n_opp >= 4:
            return -10000.0
        
        # Threat detection - opponent close to winning
        threat_penalty = 0.0
        if n_opp == 3:
            threat_penalty = -500.0
        elif n_opp >= 2:
            threat_penalty = -100.0
            
        # Opportunity bonus - we're close to winning
        opportunity_bonus = 0.0
        if n_self == 3:
            opportunity_bonus = 300.0
        elif n_self >= 2:
            opportunity_bonus = 50.0

        def accum_optimized(pieces) -> Tuple[int, int, int, int]:
            """Optimized version - only check actual pieces, not whole board"""
            mobility = 0
            blocked = 0
            reachable = 0
            control = 0
            
            for x, y, p in pieces:
                # Quick mobility estimate without full river flow calculation
                basic_moves = 0
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if (in_bounds(nx, ny, rows, cols) 
                        and not eng['is_opponent_score_cell'](nx, ny, p.owner, rows, cols, score_cols)):
                        if board[ny][nx] is None:
                            basic_moves += 1
                        elif board[ny][nx].owner != p.owner:
                            basic_moves += 0.5  # Potential push
                
                mobility += basic_moves
                if basic_moves == 0:
                    blocked += 1
                
                # Control bonus for pieces near scoring area
                if p.owner == self.player:
                    target_row = top_score_row() if p.owner == 'circle' else bottom_score_row(rows)
                    if abs(y - target_row) <= 2:
                        control += 1
                
                # Quick reachability check (simplified)
                if p.side == 'stone' and not eng['is_own_score_cell'](x, y, p.owner, rows, cols, score_cols):
                    target_row = top_score_row() if p.owner == 'circle' else bottom_score_row(rows)
                    distance_to_goal = abs(y - target_row)
                    if distance_to_goal <= 3:  # Could potentially reach in few moves
                        reachable += 1
                elif p.side == 'river' and eng['is_own_score_cell'](x, y, p.owner, rows, cols, score_cols):
                    reachable += 1
            
            return mobility, blocked, reachable, control
        mob_self, blk_self, m_self, ctrl_self = accum_optimized(my_pieces)
        mob_opp, blk_opp, m_opp, ctrl_opp = accum_optimized(opp_pieces)

        def positional(pieces) -> float:
            """Optimized positional evaluation"""
            val = 0.0
            piece_count = 0
            target = top_score_row() if self.player == 'circle' else bottom_score_row(rows)
            center = sum(score_cols) / len(score_cols)
            
            for x, y, p in pieces:
                if p.side == 'stone':
                    piece_count += 1
                    # Distance penalties
                    val += -abs(y - target) * 1.5
                    val += -abs(x - center) * 0.5
                    
                    # Bonus for stones already in scoring area
                    if eng['is_own_score_cell'](x, y, p.owner, rows, cols, score_cols):
                        val += 10.0
            
            # Game phase adjustment
            if piece_count <= self.endgame_threshold:
                val *= 1.5
            return val

        pos_self = positional(my_pieces)
        pos_opp = positional(opp_pieces)

        # Final evaluation combining all factors
        score = (self.W_SCORE * (n_self - n_opp)
                + self.W_REACH * (m_self - m_opp)
                + self.W_MOBILITY * (mob_self - mob_opp)
                + self.W_POSITION * (pos_self - pos_opp)
                + self.W_BLOCK * (blk_opp - blk_self)
                + self.W_CONTROL * (ctrl_self - ctrl_opp)
                + self.W_THREAT * threat_penalty
                + opportunity_bonus)
        
        return score

    def _quick_score(self, board, rows, cols, score_cols) -> float:
        """Enhanced quick scoring for move ordering"""
        eng = self._engine()
        top = top_score_row()
        bottom = bottom_score_row(rows)
        score = 0.0
        
        # Count pieces in scoring areas
        for x in score_cols:
            p = board[top][x]
            if p and p.side == 'stone':
                if p.owner == self.player:
                    score += 30.0
                else:
                    score -= 30.0
            q = board[bottom][x]
            if q and q.side == 'stone':
                if q.owner == self.player:
                    score += 30.0
                else:
                    score -= 30.0
        
        # Quick mobility assessment
        my_moves = 0
        opp_moves = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p:
                    continue
                info = eng['compute_valid_targets'](board, x, y, p.owner, rows, cols, score_cols)
                moves_count = len(info.get('moves', set())) + len(info.get('pushes', []))
                if p.owner == self.player:
                    my_moves += moves_count
                else:
                    opp_moves += moves_count
        
        score += (my_moves - opp_moves) * 0.1
        return score

    def _ordered_moves(self, board, player, rows, cols, score_cols) -> List[Dict[str, Any]]:
        moves = self._enumerate_moves(board, player, rows, cols, score_cols)
        random.shuffle(moves)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for mv in moves[:200]:
            ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
            if not ok:
                continue
            sc = self._quick_score(board, rows, cols, score_cols)
            self._undo_inplace(board, undo)
            scored.append((sc, mv))
        reverse = (player == self.player)
        scored.sort(key=lambda t: t[0], reverse=reverse)
        if self.BEAM_WIDTH and len(scored) > self.BEAM_WIDTH:
            scored = scored[:self.BEAM_WIDTH]
        return [mv for _, mv in scored]

    def _board_hash(self, board, rows, cols) -> str:
        """Create a hash of the board state for transposition table"""
        state = []
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p is None:
                    state.append("_")
                else:
                    state.append(f"{p.owner[0]}{p.side[0]}{p.orientation or ''}")
        return ''.join(state)

    # -------- Alpha-beta --------
    def _alphabeta(self, board, depth, alpha, beta, player, rows, cols, score_cols,
                   start_time, deadline) -> Tuple[float, Optional[Dict[str, Any]]]:
        if time.perf_counter() > deadline:
            return self._evaluate(board, rows, cols, score_cols), None
        
        # Check transposition table
        board_hash = self._board_hash(board, rows, cols)
        tt_key = (board_hash, depth, player)
        if tt_key in self.transposition_table:
            cached_score, cached_move, cached_alpha, cached_beta = self.transposition_table[tt_key]
            if cached_alpha <= alpha and cached_beta >= beta:
                return cached_score, cached_move
        
        if depth == 0:
            score = self._evaluate(board, rows, cols, score_cols)
            # Store in transposition table
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[tt_key] = (score, None, alpha, beta)
            return score, None

        maximizing = (player == self.player)
        best_move = None
        next_player = get_opponent(player)
        moves = self._ordered_moves(board, player, rows, cols, score_cols)
        if not moves:
            score = self._evaluate(board, rows, cols, score_cols)
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[tt_key] = (score, None, alpha, beta)
            return score, None

        if maximizing:
            value = float('-inf')
            for mv in moves:
                if time.perf_counter() > deadline:
                    break
                ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
                if not ok:
                    continue
                score, _ = self._alphabeta(board, depth - 1, alpha, beta, next_player,
                                            rows, cols, score_cols, start_time, deadline)
                self._undo_inplace(board, undo)
                if score > value:
                    value = score
                    best_move = mv
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break  # Beta cutoff
            
            # Store in transposition table
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[tt_key] = (value, best_move, alpha, beta)
            return value, best_move

        value = float('inf')
        for mv in moves:
            if time.perf_counter() > deadline:
                break
            ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
            if not ok:
                continue
            score, _ = self._alphabeta(board, depth - 1, alpha, beta, next_player,
                                        rows, cols, score_cols, start_time, deadline)
            self._undo_inplace(board, undo)
            if score < value:
                value = score
                best_move = mv
            if value < beta:
                beta = value
            if alpha >= beta:
                break  # Alpha cutoff
        
        # Store in transposition table
        if len(self.transposition_table) < self.max_table_size:
            self.transposition_table[tt_key] = (value, best_move, alpha, beta)
        return value, best_move

    # -------- Public choose --------
    def choose(self, board, rows, cols, score_cols, current_player_time, opponent_time):
        start = time.perf_counter()
        self.opening_moves += 1
        
        # Clear transposition table periodically to avoid memory issues
        if len(self.transposition_table) > self.max_table_size * 0.8:
            self.transposition_table.clear()
        
        # Adaptive time management
        remaining = current_player_time if current_player_time is not None else self.TIME_PER_MOVE
        
        # TEMPORARILY REMOVE TIME LIMITS FOR TESTING
        deadline = float('inf')  # No time limit
        budget = float('inf')    # No budget limit
        
        # Original time management (commented out)
        # if self.TIME_PER_MOVE is None or remaining is None:
        #     deadline = float('inf')
        # else:
        #     # Use more time in critical situations (endgame, low time)
        #     moves_remaining = max(10, 500 - self.opening_moves)  # Estimate remaining moves
        #     base_time = remaining / moves_remaining
        #     
        #     # Count pieces to detect game phase
        #     piece_count = sum(1 for y in range(rows) for x in range(cols) 
        #                     if board[y][x] and board[y][x].owner == self.player)
        #     
        #     if piece_count <= self.endgame_threshold or remaining < 10.0:
        #         # Endgame or low time - use more time per move
        #         budget = min(remaining * 0.15, base_time * 2.0)
        #     elif self.opening_moves <= 10:
        #         # Opening - use less time
        #         budget = min(base_time * 0.8, 2.0)
        #     else:
        #         # Normal play
        #         budget = min(base_time * 1.2, 4.0)
        #     
        #     budget = max(0.1, budget)  # Minimum time
        #     deadline = start + budget

        best_move = None
        max_depth = max(1, int(self.SEARCH_DEPTH))
        actual_depth_reached = 0
        
        print(f"Starting search: budget={budget:.3f}s, max_depth={max_depth}")
        
        # Iterative deepening with aspiration windows
        for depth in range(1, max_depth + 1):
            depth_start = time.perf_counter()
            if time.perf_counter() > deadline:
                print(f"Time limit reached before depth {depth}")
                break
            
            try:
                score, move = self._alphabeta(board, depth, float('-inf'), float('inf'),
                                              self.player, rows, cols, score_cols,
                                              start, deadline)
                depth_time = time.perf_counter() - depth_start
                print(f"Depth {depth}: score={score:.1f}, time={depth_time:.3f}s")
                
                if time.perf_counter() > deadline:
                    print(f"Time limit reached during depth {depth}")
                    break
                if move is not None:
                    best_move = move
                    actual_depth_reached = depth
                    
                # Early termination for very good/bad positions (DISABLED FOR TESTING)
                # if abs(score) > 1000:
                #     print(f"Early termination at depth {depth} due to decisive position (score={score})")
                #     break
                    
            except Exception as e:
                print(f"Exception at depth {depth}: {e}")
                # Fallback in case of any errors
                break
        
        total_time = time.perf_counter() - start
        print(f"Search completed: depth_reached={actual_depth_reached}, total_time={total_time:.3f}s")

        if best_move is not None:
            return best_move

        # Fallback to first valid move
        for mv in self._enumerate_moves(board, self.player, rows, cols, score_cols):
            ok, undo = self._apply_inplace(board, mv, self.player, rows, cols, score_cols)
            if ok:
                self._undo_inplace(board, undo)
                return mv
        return None

