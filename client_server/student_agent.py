import random
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

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

def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"

class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        ...

class StudentAgent(BaseAgent):
    """Alpha-beta with single-pass O(N) evaluation and solid legality alignment."""

    def __init__(self, player: str):
        super().__init__(player)

        # --- Simplified evaluation weights (compact, O(N) friendly) ---
        # Core
        self.W_SCORE = 100.0        # scored stones dominate
        self.W_PROGRESS = 2.0       # vertical progress of stones
        # Defensive structure at opponent end
        self.W_CAPS = 5.0           # our stone on opponent cap
        self.W_BLOCKER = 8.0        # blockers in opponent curtain, target 2–3
        # Mobility cues
        self.W_FLIP_POT = 2.0       # promising stone->river flips
        self.W_R_ROT_GAIN = 1.25    # rotate gain proxy
        self.W_SR_ADJ = 1.0         # stone–river adjacency readiness
        # Mild penalty for drifting behind opponent goal outside scoring files
        self.W_BEHIND_OPP = 3.0

        # Keepers near own goal
        self.K_BIG = 8.0
        self.K_SMALL = 0.0

        # Note: All other older terms (files, threats, junctions, lanes, etc.)
        # are intentionally dropped for simplicity and depth.

        # --- Search params ---
        self.SEARCH_DEPTH = 3
        self.TIME_PER_MOVE = 3.0
        self.BEAM_WIDTH = 16

        # Transposition table
        self.transposition_table = {}
        self.max_table_size = 100000

        # Track ply number for cheap phase if needed
        self.ply_count = 0
        # Small histories to avoid short cycles (store our before-turn and after-move hashes)
        self.prev_self_hashes: deque[str] = deque(maxlen=6)
        self.prev_after_hashes: deque[str] = deque(maxlen=6)
        # Track last move and recent mutation squares to avoid back-and-forth and flip/rotate spam
        self.last_my_move = None  # type: Optional[Dict[str, Any]]
        self.recent_mutations = deque(maxlen=4)  # type: deque[Tuple[str,int,int]]

        # Opening plan state
        self.opening_enabled = True
        self.opening_queue: List[Dict[str, Any]] = []
        self.OPENING_MAX_SLIDE = 3  # heuristically slide vertical rivers up to 3 cells outward
        self.opening_ride_done = {"left": False, "right": False}

        # Opening plan state
        self.opening_enabled = True
        self.opening_queue: List[Dict[str, Any]] = []

    # -------- Engine helpers --------
    def _engine(self):
        from gameEngine import (
            compute_valid_targets,
            is_own_score_cell,
            get_river_flow_destinations,
            is_opponent_score_cell,
        )
        return {
            'compute_valid_targets': compute_valid_targets,
            'is_own_score_cell': is_own_score_cell,
            'get_river_flow_destinations': get_river_flow_destinations,
            'is_opponent_score_cell': is_opponent_score_cell,
        }

    # -------- Opening (book) helpers --------
    def _front_row_y(self, rows:int) -> int:
        # Row closer to the opponent among the two starting rows
        if self.player == "circle":
            return rows - 5
        else:
            return top_score_row() + 2

    def _opening_slide_distance(self, board, rows:int, cols:int, y:int, x:int, dirx:int) -> int:
        """Heuristic distance to slide a vertical river outward along row y.
        - Count consecutive empty cells in the direction (no captures/pushes considered).
        - Cap by OPENING_MAX_SLIDE.
        """
        steps = 0
        for k in range(1, self.OPENING_MAX_SLIDE+1):
            nx = x + dirx*k
            if not in_bounds(nx, y, rows, cols):
                break
            if board[y][nx] is not None:
                break
            steps += 1
        return steps

    def _compute_opening_queue(self, board, rows, cols, score_cols):
        """Create the opening sequence V H H H H V then slide verticals outward by a heuristic distance.
        We enqueue only stone->river flips and simple horizontal moves; each step is validated lazily later.
        """
        try:
            y = self._front_row_y(rows)
            xs = [x for x in range(cols)
                  if (board[y][x] is not None and board[y][x].owner==self.player and board[y][x].side=="stone")]
            xs.sort()
            if len(xs) < 2:
                return
            plan: List[Dict[str, Any]] = []
            for i, x in enumerate(xs):
                ori = "vertical" if (i==0 or i==len(xs)-1) else "horizontal"
                plan.append({"action":"flip","from":[x,y],"orientation":ori})
            # Slide the two vertical rails outward by a heuristic number of cells (one step per queued move)
            left_x = xs[0]; right_x = xs[-1]
            k_left = self._opening_slide_distance(board, rows, cols, y, left_x, dirx=-1)
            k_right = self._opening_slide_distance(board, rows, cols, y, right_x, dirx=+1)
            # Bias toward at least 1 if possible
            for d in range(min(max(1, k_left), self.OPENING_MAX_SLIDE)):
                plan.append({"action":"move","from":[left_x - d, y],"to":[left_x - (d+1), y]})
            for d in range(min(max(1, k_right), self.OPENING_MAX_SLIDE)):
                plan.append({"action":"move","from":[right_x + d, y],"to":[right_x + (d+1), y]})
            self.opening_queue = plan
            self.opening_ride_done = {"left": False, "right": False}
        except Exception:
            self.opening_enabled = False

    def _next_opening_move(self, board, rows, cols, score_cols) -> Optional[Dict[str, Any]]:
        if not self.opening_enabled:
            return None
        if not self.opening_queue:
            self._compute_opening_queue(board, rows, cols, score_cols)
            if not self.opening_queue:
                self.opening_enabled = False
                return None
        # Only persist for the first few plies to avoid forcing bad plans
        if self.ply_count > 12:
            self.opening_enabled = False
            return None
        legal = self._enumerate_moves(board, self.player, rows, cols, score_cols)

        # Opportunistic "ride" stage: move the two inner horizontal rivers onto the vertical rails
        # and flow toward our own scoring row to open lanes.
        def try_ride(side:str) -> Optional[Dict[str, Any]]:
            if self.opening_ride_done.get(side, False):
                return None
            y = self._front_row_y(rows)
            # find nearest vertical rail and adjacent horizontal piece
            vxs = [x for x in range(cols)
                   if (board[y][x] is not None and board[y][x].owner==self.player and 
                       board[y][x].side=="river" and board[y][x].orientation=="vertical")]
            hxs = [x for x in range(cols)
                   if (board[y][x] is not None and board[y][x].owner==self.player and 
                       board[y][x].side=="river" and board[y][x].orientation=="horizontal")]
            if len(vxs) < 1 or len(hxs) < 1:
                return None
            vxs.sort(); hxs.sort()
            if side == "left":
                vx = vxs[0]
                # choose the closest horizontal to the right of vx
                hx_candidates = [hx for hx in hxs if hx > vx]
                if not hx_candidates:
                    return None
                hx = hx_candidates[0]
            else:
                vx = vxs[-1]
                # choose the closest horizontal to the left of vx
                hx_candidates = [hx for hx in hxs if hx < vx]
                if not hx_candidates:
                    return None
                hx = hx_candidates[-1]
            # Generate legal moves for from=[hx,y]
            my_goal = top_score_row() if self.player=="circle" else bottom_score_row(rows)
            best = None; best_adv = -1
            for mv in legal:
                if mv.get("action") != "move":
                    continue
                fr = mv.get("from")
                if fr != [hx, y]:
                    continue
                to = mv.get("to")
                ty = to[1]
                # improvement toward our scoring row
                # smaller distance to my_goal
                cur_dist = abs(y - my_goal)
                new_dist = abs(ty - my_goal)
                adv = cur_dist - new_dist
                # also require that the path likely flowed vertically (ty != y)
                if adv > best_adv and new_dist <= cur_dist and ty != y:
                    best_adv = adv; best = mv
            if best is not None and best_adv > 0:
                self.opening_ride_done[side] = True
                return best
            return None

        # Try ride moves before executing queued flips/moves, but only if the pattern exists
        mv_ride = try_ride("left") or try_ride("right")
        if mv_ride is not None:
            return mv_ride
        while self.opening_queue:
            target = self.opening_queue[0]
            # Drop stale entries where the source piece no longer matches
            if target["action"] == "flip":
                fx, fy = target["from"]
                p = board[fy][fx]
                if not (p and p.owner==self.player and p.side=="stone"):
                    self.opening_queue.pop(0); continue
            elif target["action"] == "move":
                fx, fy = target["from"]
                if not in_bounds(fx,fy,rows,cols) or board[fy][fx] is None or board[fy][fx].owner!=self.player:
                    self.opening_queue.pop(0); continue
            # Find matching legal move
            for mv in legal:
                if mv.get("action") != target.get("action"): continue
                if mv.get("from") != target.get("from"): continue
                if mv.get("action") == "flip":
                    if mv.get("orientation") == target.get("orientation"):
                        self.opening_queue.pop(0)
                        return mv
                elif mv.get("action") == "move":
                    if mv.get("to") == target.get("to"):
                        self.opening_queue.pop(0)
                        return mv
            # Not available now; drop and continue
            self.opening_queue.pop(0)
        self.opening_enabled = False
        return None

    # -------- Opening (book) helpers --------
    def _front_row_y(self, rows:int) -> int:
        # Row closer to the opponent among the two starting rows
        if self.player == "circle":
            return rows - 5
        else:
            return top_score_row() + 2

    def _compute_opening_queue(self, board, rows, cols, score_cols):
        try:
            y = self._front_row_y(rows)
            xs = [x for x in range(cols) if (board[y][x] is not None and board[y][x].owner==self.player and board[y][x].side=="stone")]
            xs.sort()
            if len(xs) < 2:
                return
            plan: List[Dict[str, Any]] = []
            for i, x in enumerate(xs):
                ori = "vertical" if (i==0 or i==len(xs)-1) else "horizontal"
                plan.append({"action":"flip","from":[x,y],"orientation":ori})
            left_x = xs[0]; right_x = xs[-1]
            plan.append({"action":"move","from":[left_x,y],"to":[max(0,left_x-1),y]})
            plan.append({"action":"move","from":[right_x,y],"to":[min(cols-1,right_x+1),y]})
            self.opening_queue = plan
        except Exception:
            self.opening_enabled = False

    def _next_opening_move(self, board, rows, cols, score_cols) -> Optional[Dict[str, Any]]:
        if not self.opening_enabled:
            return None
        if not self.opening_queue:
            self._compute_opening_queue(board, rows, cols, score_cols)
            if not self.opening_queue:
                self.opening_enabled = False
                return None
        if self.ply_count > 12:
            self.opening_enabled = False
            return None
        legal = self._enumerate_moves(board, self.player, rows, cols, score_cols)
        while self.opening_queue:
            target = self.opening_queue[0]
            if target["action"] == "flip":
                fx, fy = target["from"]
                p = board[fy][fx]
                if not (p and p.owner==self.player and p.side=="stone"):
                    self.opening_queue.pop(0)
                    continue
            elif target["action"] == "move":
                fx, fy = target["from"]
                p = board[fy][fx] if in_bounds(fx,fy,rows,cols) else None
                if not (p and p.owner==self.player):
                    self.opening_queue.pop(0); continue
            for mv in legal:
                if mv.get("action") != target.get("action"):
                    continue
                if mv.get("from") != target.get("from"):
                    continue
                if mv.get("action") == "flip":
                    if mv.get("orientation") == target.get("orientation"):
                        self.opening_queue.pop(0)
                        return mv
                elif mv.get("action") == "move":
                    if mv.get("to") == target.get("to"):
                        self.opening_queue.pop(0)
                        return mv
            self.opening_queue.pop(0)
        self.opening_enabled = False
        return None

    # -------- Move generation (delegate to engine, no flip/rotate bans) --------
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
                    # Only include safe flips (engine-safety): flipped river cannot allow flow into opponent score
                    for ori in ("horizontal", "vertical"):
                        prev_side, prev_ori = piece.side, piece.orientation
                        piece.side = "river"; piece.orientation = ori
                        flow = eng['get_river_flow_destinations'](board, x, y, x, y, player, rows, cols, score_cols)
                        piece.side = prev_side; piece.orientation = prev_ori
                        unsafe = False
                        for dx, dy in flow:
                            if eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols):
                                unsafe = True; break
                        if not unsafe:
                            moves.append({"action": "flip", "from": [x, y], "orientation": ori})
                else:
                    # River -> stone flip is always legal per engine
                    moves.append({"action": "flip", "from": [x, y]})
                    # Only include rotate if safe (post-rotate flow not entering opponent score)
                    prev_ori = piece.orientation
                    new_ori = "vertical" if prev_ori == "horizontal" else "horizontal"
                    piece.orientation = new_ori
                    flow = eng['get_river_flow_destinations'](board, x, y, x, y, player, rows, cols, score_cols)
                    piece.orientation = prev_ori
                    unsafe = False
                    for dx, dy in flow:
                        if eng['is_opponent_score_cell'](dx, dy, player, rows, cols, score_cols):
                            unsafe = True; break
                    if not unsafe:
                        moves.append({"action": "rotate", "from": [x, y]})
        return moves

    # -------- In-place apply / undo (uses engine generator for safety) --------
    def _apply_inplace(self, board, move, player, rows, cols, score_cols):
        # lightweight local applier for search only; assumes move came from _enumerate_moves
        action = move.get("action")
        if action in ("move","push"):
            fx,fy = move["from"]
            tx,ty = move["to"]
            piece = board[fy][fx]
            if action == "move" and board[ty][tx] is None:
                board[ty][tx] = piece; board[fy][fx] = None
                return True, {"kind":"move","fx":fx,"fy":fy,"tx":tx,"ty":ty,"m":piece}
            else:
                px,py = move["pushed_to"]
                pushed_piece = board[ty][tx]
                mover_piece = piece
                board[py][px] = pushed_piece
                board[ty][tx] = mover_piece
                board[fy][fx] = None
                flipped = False
                prev_side = mover_piece.side
                prev_ori = mover_piece.orientation
                if mover_piece.side == "river":
                    mover_piece.side = "stone"; mover_piece.orientation = None; flipped = True
                return True, {"kind":"push","fx":fx,"fy":fy,"tx":tx,"ty":ty,"px":px,"py":py,
                              "pushed":pushed_piece,"mover":mover_piece,"flipped":flipped,
                              "prev_side":prev_side,"prev_ori":prev_ori}
        elif action == "flip":
            fx,fy = move["from"]
            piece = board[fy][fx]
            if piece.side == "stone":
                piece.side = "river"; piece.orientation = move["orientation"]
                return True, {"kind":"flip","x":fx,"y":fy,"prev_side":"stone","prev_ori":None}
            else:
                prev_ori = piece.orientation
                piece.side = "stone"; piece.orientation = None
                return True, {"kind":"flip","x":fx,"y":fy,"prev_side":"river","prev_ori":prev_ori}
        elif action == "rotate":
            fx,fy = move["from"]
            piece = board[fy][fx]
            prev_ori = piece.orientation
            piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
            return True, {"kind":"rotate","x":fx,"y":fy,"prev_ori":prev_ori}
        return False, "unknown"

    def _undo_inplace(self, board, undo):
        k = undo["kind"]
        if k=="move":
            fx,fy,tx,ty = undo["fx"],undo["fy"],undo["tx"],undo["ty"]
            m = undo["m"]
            board[fy][fx]=m; board[ty][tx]=None
        elif k=="push":
            fx,fy,tx,ty,px,py = undo["fx"],undo["fy"],undo["tx"],undo["ty"],undo["px"],undo["py"]
            mover = undo["mover"]; pushed = undo["pushed"]
            if undo["flipped"]:
                mover.side = undo["prev_side"]; mover.orientation = undo["prev_ori"]
            board[fy][fx]=mover; board[ty][tx]=pushed; board[py][px]=None
        elif k=="flip":
            x,y = undo["x"],undo["y"]
            piece = board[y][x]
            if undo["prev_side"]=="stone":
                piece.side="stone"; piece.orientation=None
            else:
                piece.side="river"; piece.orientation=undo["prev_ori"]
        elif k=="rotate":
            x,y = undo["x"],undo["y"]
            piece = board[y][x]
            piece.orientation = undo["prev_ori"]

    # -------- O(N) Evaluation (single pass; simplified) --------
    def _evaluate_simple(self, board, rows, cols, score_cols) -> float:
        eng = self._engine()
        my_goal = top_score_row() if self.player == "circle" else bottom_score_row(rows)
        opp_goal = top_score_row() if self.opponent == "circle" else bottom_score_row(rows)

        # Accumulators
        n_self = n_opp = 0
        progress_self = progress_opp = 0
        caps_self = caps_opp = 0
        blockers_self = 0
        flip_potential_sum = 0.0
        rotate_gain_self = 0
        sr_adj_self = sr_adj_opp = 0
        keepers_self = keepers_opp = 0
        behind_opp_penalty_self = 0.0

        def is_score_cell_for(pl, x, y):
            return eng['is_own_score_cell'](x, y, pl, rows, cols, score_cols)

        def span_along(x, y, horizontal: bool, radius: int) -> int:
            total = 0
            if horizontal:
                for dx in (-1, -2) if radius >= 2 else (-1,):
                    nx = x + dx
                    if not in_bounds(nx, y, rows, cols): break
                    q = board[y][nx]
                    if q is None or getattr(q, 'side', 'stone') == "river": total += 1
                    else: break
                for dx in (1, 2) if radius >= 2 else (1,):
                    nx = x + dx
                    if not in_bounds(nx, y, rows, cols): break
                    q = board[y][nx]
                    if q is None or getattr(q, 'side', 'stone') == "river": total += 1
                    else: break
            else:
                for dy in (-1, -2) if radius >= 2 else (-1,):
                    ny = y + dy
                    if not in_bounds(x, ny, rows, cols): break
                    q = board[ny][x]
                    if q is None or getattr(q, 'side', 'stone') == "river": total += 1
                    else: break
                for dy in (1, 2) if radius >= 2 else (1,):
                    ny = y + dy
                    if not in_bounds(x, ny, rows, cols): break
                    q = board[ny][x]
                    if q is None or getattr(q, 'side', 'stone') == "river": total += 1
                    else: break
            return total

        curtain_rows_opp = (range(opp_goal-2, opp_goal) if self.opponent == "circle" else range(opp_goal+1, opp_goal+3))
        cap_y_opp = opp_goal - 1 if self.opponent == "circle" else opp_goal + 1
        cap_y_self = my_goal - 1 if self.player == "circle" else my_goal + 1

        keeper_rows_self = (range(my_goal, my_goal+3) if self.player == "circle" else range(my_goal-2, my_goal+1))
        keeper_rows_opp = (range(opp_goal, opp_goal+3) if self.opponent == "circle" else range(opp_goal-2, opp_goal+1))

        for y in range(rows):
            for x in range(cols):
                p = board[y][x]

                if x in score_cols and y in curtain_rows_opp:
                    if p and p.owner == self.player and p.side == "stone":
                        blockers_self += 1

                if p is None:
                    continue

                if p.side == "stone":
                    if p.owner == self.player and y in keeper_rows_self:
                        keepers_self += 1
                    if p.owner == self.opponent and y in keeper_rows_opp:
                        keepers_opp += 1

                if p.side == "stone" and is_score_cell_for(p.owner, x, y):
                    if p.owner == self.player: n_self += 1
                    else: n_opp += 1

                if p.side == "stone":
                    goal = top_score_row() if p.owner == "circle" else bottom_score_row(rows)
                    dist = abs(y - goal)
                    maxd = max(abs(0 - goal), abs((rows-1) - goal))
                    prog = (maxd - dist)
                    if p.owner == self.player: progress_self += prog
                    else: progress_opp += prog

                if x in score_cols and y == cap_y_opp and p.side == "stone" and p.owner == self.player:
                    caps_self += 1
                if x in score_cols and y == cap_y_self and p.side == "stone" and p.owner == self.opponent:
                    caps_opp += 1

                if p.owner == self.player and p.side == "stone":
                    hspan = span_along(x, y, True, 2)
                    vspan = span_along(x, y, False, 2)
                    best = max(hspan, vspan)
                    if best >= 2: flip_potential_sum += 1.0
                    elif best >= 1: flip_potential_sum += 0.5
                    # Penalty: our stone beyond opponent goal and outside score files (can't block/score)
                    if (self.player == "circle" and y > opp_goal and x not in score_cols) or (self.player == "square" and y < opp_goal and x not in score_cols):
                        behind_opp_penalty_self += 1.0

                if p.side == "stone":
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if not in_bounds(nx, ny, rows, cols):
                            continue
                        q = board[ny][nx]
                        if q and q.side == "river":
                            if q.orientation == "horizontal":
                                nx2 = nx + (1 if dx>=0 else -1)
                                ok = in_bounds(nx2, ny, rows, cols) and (board[ny][nx2] is None or getattr(board[ny][nx2],'side','stone')=="river")
                            else:
                                ny2 = ny + (1 if dy>=0 else -1)
                                ok = in_bounds(nx, ny2, rows, cols) and (board[ny2][nx] is None or getattr(board[ny2][nx],'side','stone')=="river")
                            if ok:
                                if p.owner == self.player and q.owner == self.player:
                                    sr_adj_self += 1; break
                                if p.owner == self.opponent and q.owner == self.opponent:
                                    sr_adj_opp += 1; break

                if p.owner == self.player and p.side == "river":
                    horiz = (p.orientation == "horizontal")
                    span_curr = span_along(x, y, horiz, 1)
                    span_rot = span_along(x, y, not horiz, 1)
                    if span_rot > span_curr:
                        rotate_gain_self += 1

        score = 0.0
        score += self.W_SCORE * (n_self - n_opp)
        score += self.W_PROGRESS * (progress_self - progress_opp)
        score += self.W_CAPS * (caps_self - caps_opp)
        score += self.W_BLOCKER * (min(blockers_self, 3) - 2)
        score += self.W_FLIP_POT * (flip_potential_sum)
        score += self.W_R_ROT_GAIN * (rotate_gain_self)
        score += self.W_SR_ADJ * (sr_adj_self - sr_adj_opp)
        score -= self.W_BEHIND_OPP * behind_opp_penalty_self

        def keep_score(k):
            if k <= 0: return -self.K_BIG
            if k == 1: return -self.K_SMALL
            return self.K_SMALL
        score += keep_score(keepers_self)
        score -= keep_score(keepers_opp)

        if n_self >= 4: return 10000.0
        if n_opp >= 4: return -10000.0
        return score
    # -------- O(N) Evaluation (single pass; no flow) --------
    def _evaluate(self, board, rows, cols, score_cols) -> float:
        eng = self._engine()

        # Precompute constants
        my_goal = top_score_row() if self.player=="circle" else bottom_score_row(rows)
        opp_goal = top_score_row() if self.opponent=="circle" else bottom_score_row(rows)
        center_x = sum(score_cols)/len(score_cols)

        # Accumulators
        n_self = n_opp = 0
        progress_self = progress_opp = 0
        # Cap/cap-like metrics
        caps_self = caps_opp = 0
        cap_horiz_riv_self = 0
        cap_bad_for_us = 0  # empty cap or enemy vertical river at opp caps
        # Curtain and blockers
        curtain_self = curtain_opp = 0
        blockers_self = 0
        # Junctions
        junction_self = junction_opp = 0
        # Threats
        threats_opp = threats_self = 0
        # Anchors
        antipush_self = antipush_opp = 0
        # Safe rivers near goals
        saferiv_self = saferiv_opp = 0
        # Files presence
        file_presence_self = file_presence_opp = 0
        # Mobility proxies
        flip_potential_sum = 0.0
        river_span_self = river_span_opp = 0
        rotate_gain_self = 0
        sr_adj_self = sr_adj_opp = 0
        # Keepers
        keepers_self = 0
        keepers_opp = 0
        # Behind-goal penalties
        behind_penalty_self = 0.0
        # Lane openness & push threats near opponent goal (defense)
        lane_open_opp = 0.0
        riv_push_threat_opp = 0
        side_push_threat_opp = 0
        # Home-side bonuses and penalties
        home_caps_self = 0
        home_curtain_self = 0
        behind_own_penalty_self = 0.0

        # Helpers
        def is_score_cell_for(pl, x, y):
            return eng['is_own_score_cell'](x,y,pl,rows,cols,score_cols)

        def forward_row(pl, y):
            return y - 1 if pl=="circle" else y + 1

        # Helper: count span along an orientation from (x,y), up to given radius
        def span_along(x, y, horizontal: bool, radius: int) -> int:
            total = 0
            if horizontal:
                # left
                for dx in ( -1, -2 ) if radius >= 2 else (-1,):
                    nx = x + dx
                    if not in_bounds(nx, y, rows, cols): break
                    if eng['is_own_score_cell'](nx, y, self.opponent, rows, cols, score_cols): break  # opponent score forbidden
                    q = board[y][nx]
                    if q is None or q.side=="river": total += 1
                    else: break
                # right
                for dx in ( 1, 2 ) if radius >= 2 else (1,):
                    nx = x + dx
                    if not in_bounds(nx, y, rows, cols): break
                    if eng['is_own_score_cell'](nx, y, self.opponent, rows, cols, score_cols): break
                    q = board[y][nx]
                    if q is None or q.side=="river": total += 1
                    else: break
            else:
                # up
                for dy in ( -1, -2 ) if radius >= 2 else (-1,):
                    ny = y + dy
                    if not in_bounds(x, ny, rows, cols): break
                    if eng['is_own_score_cell'](x, ny, self.opponent, rows, cols, score_cols): break
                    q = board[ny][x]
                    if q is None or q.side=="river": total += 1
                    else: break
                # down
                for dy in ( 1, 2 ) if radius >= 2 else (1,):
                    ny = y + dy
                    if not in_bounds(x, ny, rows, cols): break
                    if eng['is_own_score_cell'](x, ny, self.opponent, rows, cols, score_cols): break
                    q = board[ny][x]
                    if q is None or q.side=="river": total += 1
                    else: break
            return total

        # Helper: junction around cell (at least one vertical river N/S and one horizontal river E/W)
        def has_junction_around(x, y) -> bool:
            vert = False; horiz = False
            # N/S
            for dy in (-1, 1):
                ny = y + dy
                if in_bounds(x, ny, rows, cols):
                    q = board[ny][x]
                    if q and q.side=="river" and q.orientation=="vertical":
                        vert = True; break
            # E/W
            for dx in (-1, 1):
                nx = x + dx
                if in_bounds(nx, y, rows, cols):
                    q = board[y][nx]
                    if q and q.side=="river" and q.orientation=="horizontal":
                        horiz = True; break
            return vert and horiz

        # Fast line checks for caps/curtains
        cap_y_opp = opp_goal - 1 if self.opponent=="circle" else opp_goal + 1
        curtain_rows_opp = (
            range(opp_goal-2, opp_goal) if self.opponent=="circle" else range(opp_goal+1, opp_goal+3)
        )

        cap_y_self = my_goal - 1 if self.player=="circle" else my_goal + 1
        curtain_rows_self = (
            range(my_goal-2, my_goal) if self.player=="circle" else range(my_goal+1, my_goal+3)
        )

        # Bands for keepers
        keeper_rows_self = (range(my_goal, my_goal+3) if self.player=="circle" else range(my_goal-2, my_goal+1))
        keeper_rows_opp = (range(opp_goal, opp_goal+3) if self.opponent=="circle" else range(opp_goal-2, opp_goal+1))

        # Precompute step signs relative to opponent goal
        step_to_opp_goal = -1 if self.opponent=="circle" else 1
        step_from_opp_goal = 1 if self.opponent=="circle" else -1

        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                # score-file presence (stones only makes sense, but cheap to count any presence)
                if x in score_cols:
                    if p and p.owner==self.player: file_presence_self += 1
                    elif p and p.owner==self.opponent: file_presence_opp += 1

                # blockers in opponent curtain (stones only)
                if x in score_cols and y in curtain_rows_opp:
                    if p and p.owner==self.player and p.side=="stone":
                        blockers_self += 1

                # junctions near opponent goal
                if x in score_cols and (
                    (self.opponent=="circle" and opp_goal-3 <= y <= opp_goal-1) or
                    (self.opponent!="circle" and opp_goal+1 <= y <= opp_goal+3)
                ):
                    if has_junction_around(x, y):
                        if p and p.side=="stone":
                            if p.owner==self.player: junction_self += 1
                            else: junction_opp += 1

                # keepers near own goal
                if p and p.side=="stone":
                    if y in keeper_rows_self and p.owner==self.player:
                        keepers_self += 1
                    if y in keeper_rows_opp and p.owner==self.opponent:
                        keepers_opp += 1

                if not p:
                    # safe rivers metric handled when we see rivers
                    continue

                # Scored stones
                if p.side=="stone":
                    if is_score_cell_for(p.owner, x, y):
                        if p.owner==self.player: n_self += 1
                        else: n_opp += 1

                # Vertical progress (stones only)
                if p.side=="stone":
                    goal = top_score_row() if p.owner=="circle" else bottom_score_row(rows)
                    dist = abs(y - goal)
                    maxd = max(abs(0 - goal), abs((rows-1) - goal))
                    prog = (maxd - dist)
                    if p.owner==self.player: progress_self += prog
                    else: progress_opp += prog

                # Caps and curtains (only count once per file/row pair)
                if x in score_cols:
                    # Opponent caps/curtain we want to block
                    if y == cap_y_opp:
                        if p.owner==self.player and p.side=="stone":
                            caps_self += 1
                        elif p.owner==self.player and p.side=="river" and p.orientation=="horizontal":
                            cap_horiz_riv_self += 1
                        elif p is None or (p.owner==self.opponent and p.side=="river" and p.orientation=="vertical"):
                            cap_bad_for_us += 1

                    if y in curtain_rows_opp:
                        if p.owner==self.player and p.side=="stone":
                            curtain_self += 1

                    # Our own end (mirror, small influence + explicit home rewards)
                    if y == cap_y_self:
                        if p.owner==self.opponent and p.side=="stone":
                            caps_opp += 1
                        if p.owner==self.player and p.side=="stone":
                            home_caps_self += 1
                    if y in curtain_rows_self:
                        if p.owner==self.opponent and p.side=="stone":
                            curtain_opp += 1
                        if p.owner==self.player and p.side=="stone":
                            home_curtain_self += 1

                # Front-runner threat (enemy stones within 2 rows of their goal with open forward cell)
                if p.owner==self.opponent and p.side=="stone":
                    goal = opp_goal
                    within2 = (abs(y - goal) <= 2)
                    if within2:
                        fy = y-1 if self.opponent=="circle" else y+1
                        if in_bounds(x,fy,rows,cols):
                            q = board[fy][x]
                            if q is None or (q.side=="river"):  # empty or river => live lane
                                threats_opp += 1
                            elif q.owner==self.player and q.side=="stone":
                                # we block their step
                                pass

                # Anti-push anchors on opponent caps (protect our blocks)
                if p.owner==self.player and p.side=="stone" and x in score_cols and y == cap_y_opp:
                    # check if trivial 1-step push is possible: look behind our cap in enemy direction
                    bdx, bdy = (0, -1) if self.opponent=="circle" else (0, 1)
                    bx, by = x+bdx, y+bdy
                    fdx, fdy = (0, 1) if self.opponent=="circle" else (0, -1)  # direction a pusher would come from
                    px, py = x+fdx, y+fdy
                    if in_bounds(px,py,rows,cols) and in_bounds(bx,by,rows,cols):
                        pusher = board[py][px]
                        behind = board[by][bx]
                        if not (pusher and pusher.owner==self.opponent and behind is None):
                            antipush_self += 1

                # Opponent anchors on our caps (hurts us)
                if p.owner==self.opponent and p.side=="stone" and x in score_cols and y == cap_y_self:
                    bdx, bdy = (0, -1) if self.player=="circle" else (0, 1)
                    bx, by = x+bdx, y+bdy
                    fdx, fdy = (0, 1) if self.player=="circle" else (0, -1)
                    px, py = x+fdx, y+fdy
                    if in_bounds(px,py,rows,cols) and in_bounds(bx,by,rows,cols):
                        pusher = board[py][px]
                        behind = board[by][bx]
                        if not (pusher and pusher.owner==self.player and behind is None):
                            antipush_opp += 1

                # Safe rivers near goals (prefer horizontal near opp goal; penalize enemy horizontals near our goal less)
                if p.owner==self.player and p.side=="river":
                    band = range(opp_goal-3, opp_goal) if self.opponent=="circle" else range(opp_goal+1, opp_goal+4)
                    if y in band and p.orientation=="horizontal":
                        saferiv_self += 1
                if p.owner==self.opponent and p.side=="river":
                    band = range(my_goal-3, my_goal) if self.player=="circle" else range(my_goal+1, my_goal+4)
                    if y in band and p.orientation=="horizontal":
                        saferiv_opp += 1

                # Mobility proxies
                if p.owner==self.player and p.side=="stone":
                    # Flip potential (radius=2 both orientations)
                    hspan = span_along(x, y, True, 2)
                    vspan = span_along(x, y, False, 2)
                    best = max(hspan, vspan)
                    if best >= 2:
                        flip_potential_sum += 1.0
                    elif best >= 1:
                        flip_potential_sum += 0.5
                    # Behind-opponent-goal penalty: if our stone is beyond their goal row and not in score files
                    if (self.player=="circle" and y < opp_goal and x not in score_cols) or \
                       (self.player=="square" and y > opp_goal and x not in score_cols):
                        # If opponent stones within 2 squares toward their goal, stronger penalty
                        step = 1 if self.player=="circle" else -1
                        blocked = False
                        for k in (1,2):
                            ny = y + k*step
                            if not in_bounds(x, ny, rows, cols):
                                break
                            q2 = board[ny][x]
                            if q2 and q2.owner==self.opponent and q2.side=="stone":
                                blocked = True; break
                        behind_penalty_self += (1.5 if blocked else 1.0)
                    # Behind-own-goal penalty: if our stone is beyond our goal row and not in score files
                    if (self.player=="circle" and y < my_goal and x not in score_cols) or \
                       (self.player=="square" and y > my_goal and x not in score_cols):
                        behind_own_penalty_self += 1.0
                    # Stone–river adjacency readiness
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if not in_bounds(nx,ny,rows,cols):
                            continue
                        q = board[ny][nx]
                        if q and q.owner==self.player and q.side=="river":
                            # if next step along river orientation from that river cell is clear (empty/river), it's rideable soon
                            if q.orientation=="horizontal":
                                nx2 = nx + (1 if dx>=0 else -1)
                                if in_bounds(nx2, ny, rows, cols):
                                    r2 = board[ny][nx2]
                                    if r2 is None or (r2.side=="river"):
                                        sr_adj_self += 1; break
                            else:
                                ny2 = ny + (1 if dy>=0 else -1)
                                if in_bounds(nx, ny2, rows, cols):
                                    r2 = board[ny2][nx]
                                    if r2 is None or (r2.side=="river"):
                                        sr_adj_self += 1; break

                if p.owner==self.opponent and p.side=="stone":
                    # Opp symmetry for adjacency (hurts us)
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if not in_bounds(nx,ny,rows,cols):
                            continue
                        q = board[ny][nx]
                        if q and q.owner==self.opponent and q.side=="river":
                            if q.orientation=="horizontal":
                                nx2 = nx + (1 if dx>=0 else -1)
                                if in_bounds(nx2, ny, rows, cols):
                                    r2 = board[ny][nx2]
                                    if r2 is None or (r2.side=="river"):
                                        sr_adj_opp += 1; break
                            else:
                                ny2 = ny + (1 if dy>=0 else -1)
                                if in_bounds(nx, ny2, rows, cols):
                                    r2 = board[ny2][nx]
                                    if r2 is None or (r2.side=="river"):
                                        sr_adj_opp += 1; break

                if p.side=="river":
                    # River span along its orientation (radius=2)
                    horiz = (p.orientation=="horizontal")
                    sp = span_along(x, y, horiz, 2)
                    if p.owner==self.player:
                        river_span_self += sp
                        # Rotate-gain proxy: compare immediate neighbors (radius=1) current vs rotated
                        span1_curr = span_along(x, y, horiz, 1)
                        span1_rot = span_along(x, y, not horiz, 1)
                        if span1_rot > span1_curr:
                            rotate_gain_self += 1
                    else:
                        river_span_opp += sp

        # After scan: evaluate lane openness and push threats per score-file (opponent's approach)
        for x in score_cols:
            capy = cap_y_opp
            if not in_bounds(x, capy, rows, cols):
                continue
            cap_cell = board[capy][x]
            goal_y = capy + step_to_opp_goal  # this is opp_goal
            out1_y = capy + step_from_opp_goal
            out2_y = capy + 2*step_from_opp_goal

            # Open lane scoring: penalize empty/vertical-river chain just outside cap
            lane_score = 0.0
            if cap_cell is None:
                lane_score += 1.0
            elif cap_cell.owner == self.opponent and cap_cell.side=="river" and cap_cell.orientation=="vertical":
                lane_score += 2.0

            if in_bounds(x, out1_y, rows, cols):
                c1 = board[out1_y][x]
                if c1 is None:
                    lane_score += 0.5
                elif c1.owner==self.opponent and c1.side=="river" and c1.orientation=="vertical":
                    lane_score += 1.0
            if in_bounds(x, out2_y, rows, cols):
                c2 = board[out2_y][x]
                if c2 and c2.owner==self.opponent and c2.side=="river" and c2.orientation=="vertical":
                    lane_score += 0.5
            lane_open_opp += lane_score

            # River-push threat against our cap stone: enemy river directly outside cap along vertical
            if cap_cell and cap_cell.owner==self.player and cap_cell.side=="stone":
                # if scoring cell (goal_y) is empty, river or friendly river-to-stone flip would allow push into goal
                if in_bounds(x, goal_y, rows, cols):
                    goal_cell = board[goal_y][x]
                    if goal_cell is None:
                        if in_bounds(x, out1_y, rows, cols):
                            c1 = board[out1_y][x]
                            if c1 and c1.owner==self.opponent and c1.side=="river" and c1.orientation=="vertical":
                                riv_push_threat_opp += 1

                # Side-push threat: enemy stone at left/right with empty pushed_to on opposite side
                for dx in (-1, 1):
                    nx = x + dx
                    px = x - dx  # pushed_to cell on the other side
                    if not in_bounds(nx, capy, rows, cols) or not in_bounds(px, capy, rows, cols):
                        continue
                    neigh = board[capy][nx]
                    pushed_to = board[capy][px]
                    if neigh and neigh.owner==self.opponent and neigh.side=="stone" and pushed_to is None:
                        side_push_threat_opp += 1

        # Final linear eval (bounded, cheap)
        score = 0.0
        score += self.W_SCORE * (n_self - n_opp)
        score += self.W_PROGRESS * (progress_self - progress_opp)
        # Defensive/caps
        score += self.W_CAPS * ((caps_self + 0.5*cap_horiz_riv_self) - caps_opp - cap_bad_for_us)
        # Curtain and blockers
        score += self.W_CURTAIN * (curtain_self - curtain_opp)
        # Blocker target curve toward 2–3
        score += self.W_BLOCKER * (min(blockers_self, 3) - 2)
        # Junction chokes
        score += self.W_JUNCTION * (junction_self - junction_opp)
        # Threats and anchors
        score += self.W_THREAT * (threats_opp - threats_self)
        score += self.W_ANTIPUSH * (antipush_self - antipush_opp)
        # Mobility proxies
        score += self.W_FLIP_POT * (flip_potential_sum)
        score += self.W_RIVER_SPAN * (river_span_self - river_span_opp)
        score += self.W_R_ROT_GAIN * (rotate_gain_self)
        score += self.W_SR_ADJ * (sr_adj_self - sr_adj_opp)
        # Safe rivers near goal
        score += self.W_SAFE_RIV * (saferiv_self - saferiv_opp)
        # Open lanes and push threats near opponent goal (defense)
        score -= self.W_LANE * (lane_open_opp)
        score -= self.W_RIV_PUSH_TH * (riv_push_threat_opp)
        score -= self.W_SIDE_PUSH_TH * (side_push_threat_opp)
        # File presence
        score += self.W_FILES * (file_presence_self - file_presence_opp)
        # Behind-goal penalty
        score -= self.W_BEHIND * (behind_penalty_self)
        # Home-side bonuses and own-behind penalty
        score += self.W_HOME_CAP * home_caps_self
        score += self.W_HOME_CURTAIN * home_curtain_self
        score -= self.W_BEHIND_OWN * behind_own_penalty_self

        # Keepers scoring (piecewise)
        def keep_score(k):
            if k <= 0: return -self.K_BIG
            if k == 1: return -self.K_SMALL
            return self.K_SMALL
        score += keep_score(keepers_self)
        score -= keep_score(keepers_opp)  # opponent having keepers hurts us slightly

        # Terminal shortcuts
        if n_self >= 4: return 10000.0
        if n_opp >= 4: return -10000.0
        return score

    # -------- Simple quick-score for ordering (O(N), subset of eval) --------
    def _quick_score(self, board, rows, cols, score_cols) -> float:
        # Cheap: scored stones + caps + rough blockers
        top = top_score_row(); bot = bottom_score_row(rows)
        n_self = n_opp = 0
        caps_self = caps_opp = 0
        cap_bad = 0
        blockers = 0
        opp_goal = top if self.opponent=="circle" else bot
        cap_y_opp = opp_goal - 1 if self.opponent=="circle" else opp_goal + 1
        curtain_rows_opp = (range(opp_goal-2, opp_goal) if self.opponent=="circle" else range(opp_goal+1, opp_goal+2+1))
        for x in score_cols:
            p = board[top][x]
            if p and p.side=="stone": n_self += (1 if p.owner==self.player else 0); n_opp += (1 if p.owner==self.opponent else 0)
            q = board[bot][x]
            if q and q.side=="stone": n_self += (1 if q.owner==self.player else 0); n_opp += (1 if q.owner==self.opponent else 0)
        for x in score_cols:
            p = board[cap_y_opp][x] if in_bounds(x,cap_y_opp,rows,cols) else None
            if p is None or (p and p.owner==self.opponent and p.side=="river" and p.orientation=="vertical"): cap_bad += 1
            if p and p.owner==self.player and p.side=="stone": caps_self += 1
            if p and p.owner==self.opponent and p.side=="stone": caps_opp += 1
        # blockers in opponent curtain
        for y in curtain_rows_opp:
            for x in score_cols:
                p = board[y][x]
                if p and p.owner==self.player and p.side=="stone": blockers += 1
        blocker_term = (min(blockers,3) - 2)
        return 100.0*(n_self-n_opp) + 5.0*(caps_self-caps_opp) - 5.0*cap_bad + 8.0*blocker_term

    # -------- Move ordering --------
    def _ordered_moves(self, board, player, rows, cols, score_cols) -> List[Dict[str, Any]]:
        moves = self._enumerate_moves(board, player, rows, cols, score_cols)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        def local_flip_potential(x:int,y:int) -> float:
            # Use same cheap spans (radius=2)
            # Only consider own stones (when ordering for self)
            p = board[y][x]
            if not p or p.owner != player or p.side != "stone":
                return 0.0
            # quick local span
            def span1d(horizontal: bool, r: int) -> int:
                total = 0
                if horizontal:
                    for dx in (-1, -2):
                        nx = x + dx
                        if not in_bounds(nx,y,rows,cols): break
                        q = board[y][nx]
                        if q is None or q.side=="river": total += 1
                        else: break
                    for dx in (1, 2):
                        nx = x + dx
                        if not in_bounds(nx,y,rows,cols): break
                        q = board[y][nx]
                        if q is None or q.side=="river": total += 1
                        else: break
                else:
                    for dy in (-1, -2):
                        ny = y + dy
                        if not in_bounds(x,ny,rows,cols): break
                        q = board[ny][x]
                        if q is None or q.side=="river": total += 1
                        else: break
                    for dy in (1, 2):
                        ny = y + dy
                        if not in_bounds(x,ny,rows,cols): break
                        q = board[ny][x]
                        if q is None or q.side=="river": total += 1
                        else: break
                return total
            h = span1d(True, 2); v = span1d(False, 2)
            best = max(h,v)
            return 1.0 if best >= 2 else (0.5 if best >= 1 else 0.0)

        def local_rotate_gain(x:int,y:int) -> float:
            p = board[y][x]
            if not p or p.owner!=player or p.side!="river":
                return 0.0
            horiz = (p.orientation=="horizontal")
            # immediate neighbors count
            def span1(horizontal: bool) -> int:
                cnt = 0
                if horizontal:
                    for dx in (-1,1):
                        nx = x + dx
                        if not in_bounds(nx,y,rows,cols): continue
                        q = board[y][nx]
                        if q is None or q.side=="river": cnt += 1
                else:
                    for dy in (-1,1):
                        ny = y + dy
                        if not in_bounds(x,ny,rows,cols): continue
                        q = board[ny][x]
                        if q is None or q.side=="river": cnt += 1
                return cnt
            curr = span1(horiz); rot = span1(not horiz)
            return 1.0 if rot > curr else 0.0

        limit = 160 if player == self.player else 120  # search our moves a bit broader
        for mv in moves[:limit]:
            # Local priority bump for flips/rotates that look promising
            prio = 0.0
            if mv.get("action") == "flip":
                fx, fy = mv["from"]
                # only prioritize stone->river flips; river->stone is situational
                p = board[fy][fx]
                if p and p.owner==player and p.side=="stone":
                    pot = local_flip_potential(fx, fy)
                    prio += 4.0 * pot
                # demote spam of flip on same square as our recent mutation
                if player == self.player:
                    for act, mx, my in list(self.recent_mutations):
                        if act == "flip" and (mx, my) == (fx, fy):
                            prio -= 6.0; break
            elif mv.get("action") == "rotate":
                fx, fy = mv["from"]
                p = board[fy][fx]
                if p and p.owner==player and p.side=="river":
                    gain = local_rotate_gain(fx, fy)
                    prio += 3.0 * gain
                    # extra if near opponent goal and rotating vertical->horizontal
                    opp_goal = top_score_row() if (player=="square") else bottom_score_row(rows)
                    band = (range(opp_goal-3, opp_goal) if (player=="square") else range(opp_goal+1, opp_goal+4))
                    if fy in band and p.orientation=="vertical":
                        prio += 1.0
                # demote spam of rotate on same square as our recent mutation
                if player == self.player:
                    for act, mx, my in list(self.recent_mutations):
                        if act == "rotate" and (mx, my) == (fx, fy):
                            prio -= 8.0; break
            elif mv.get("action") == "move":
                # Demote moves that go behind the opponent's goal outside score files (for that player)
                tx, ty = mv["to"]
                opp_goal_local = top_score_row() if player=="square" else bottom_score_row(rows)
                # For circle: behind opp goal means y > opp_goal; for square: y < opp_goal
                if (player=="circle" and ty > opp_goal_local and tx not in score_cols) or \
                   (player=="square" and ty < opp_goal_local and tx not in score_cols):
                    prio -= 6.0
                # Demote moves that go behind own goal outside score files (weaker)
                my_goal_local = top_score_row() if player=="circle" else bottom_score_row(rows)
                if (player=="circle" and ty < my_goal_local and tx not in score_cols) or \
                   (player=="square" and ty > my_goal_local and tx not in score_cols):
                    prio -= 3.0

            # Back-and-forth demotion relative to our last move (only for our own ordering)
            if player == self.player and self.last_my_move is not None:
                lm = self.last_my_move
                if mv.get("action") == "move" and lm.get("action") == "move":
                    if mv.get("from") == lm.get("to") and mv.get("to") == lm.get("from"):
                        prio -= 10.0
                if mv.get("action") == "rotate" and lm.get("action") == "rotate":
                    if mv.get("from") == lm.get("from"):
                        prio -= 8.0
                if mv.get("action") == "flip" and lm.get("action") == "flip":
                    if mv.get("from") == lm.get("from"):
                        prio -= 6.0

            ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
            if not ok:
                continue
            sc = self._quick_score(board, rows, cols, score_cols) + prio
            # Repetition guard: demote moves that return to any of our recent hashes (before-turn or after-move)
            if player == self.player and (self.prev_self_hashes or self.prev_after_hashes):
                new_hash = self._board_hash(board, rows, cols)
                if (new_hash in self.prev_self_hashes) or (new_hash in self.prev_after_hashes):
                    sc -= 50.0
            self._undo_inplace(board, undo)
            scored.append((sc, mv))
        reverse = (player == self.player)
        scored.sort(key=lambda t: t[0], reverse=reverse)
        if self.BEAM_WIDTH and len(scored) > self.BEAM_WIDTH:
            scored = scored[:self.BEAM_WIDTH]
        return [mv for _, mv in scored]

    # -------- TT hash --------
    def _board_hash(self, board, rows, cols) -> str:
        state=[]
        for y in range(rows):
            for x in range(cols):
                p=board[y][x]
                if p is None: state.append("_")
                else: state.append(f"{p.owner[0]}{p.side[0]}{p.orientation[0] if p.orientation else '-'}")
        return "".join(state)

    # -------- Alpha-beta --------
    def _alphabeta(self, board, depth, alpha, beta, player, rows, cols, score_cols,
                   start_time, deadline) -> Tuple[float, Optional[Dict[str, Any]]]:
        if time.perf_counter() > deadline:
            return self._evaluate_simple(board, rows, cols, score_cols), None

        key = (self._board_hash(board,rows,cols), depth, player)
        if key in self.transposition_table:
            s, mv, da, db = self.transposition_table[key]
            # simple reuse with repetition guard on our turn
            if mv is not None and player == self.player and self.prev_self_hashes:
                ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
                if ok:
                    new_hash = self._board_hash(board, rows, cols)
                    self._undo_inplace(board, undo)
                    if new_hash in self.prev_self_hashes:
                        pass  # ignore TT entry to avoid cycling move
                    else:
                        return s, mv
                else:
                    return s, mv
            else:
                return s, mv

        if depth == 0:
            s = self._evaluate_simple(board, rows, cols, score_cols)
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[key] = (s, None, alpha, beta)
            return s, None

        moves = self._ordered_moves(board, player, rows, cols, score_cols)
        if not moves:
            s = self._evaluate_simple(board, rows, cols, score_cols)
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[key] = (s, None, alpha, beta)
            return s, None

        maximizing = (player == self.player)
        best_move = None

        if maximizing:
            v = float('-inf')
            for mv in moves:
                if time.perf_counter() > deadline: break
                ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
                if not ok: continue
                s, _ = self._alphabeta(board, depth-1, alpha, beta, get_opponent(player),
                                       rows, cols, score_cols, start_time, deadline)
                self._undo_inplace(board, undo)
                if s > v: v, best_move = s, mv
                if v > alpha: alpha = v
                if alpha >= beta: break
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[key] = (v, best_move, alpha, beta)
            return v, best_move
        else:
            v = float('inf')
            for mv in moves:
                if time.perf_counter() > deadline: break
                ok, undo = self._apply_inplace(board, mv, player, rows, cols, score_cols)
                if not ok: continue
                s, _ = self._alphabeta(board, depth-1, alpha, beta, get_opponent(player),
                                       rows, cols, score_cols, start_time, deadline)
                self._undo_inplace(board, undo)
                if s < v: v, best_move = s, mv
                if v < beta: beta = v
                if alpha >= beta: break
            if len(self.transposition_table) < self.max_table_size:
                self.transposition_table[key] = (v, best_move, alpha, beta)
            return v, best_move

    # -------- Public choose --------
    def choose(self, board, rows, cols, score_cols, current_player_time, opponent_time):
        start = time.perf_counter()
        self.ply_count += 1

        if len(self.transposition_table) > self.max_table_size * 0.8:
            self.transposition_table.clear()

        # (You can re-enable a strict deadline if desired)
        deadline = float('inf')

        # Opening move attempt (short-circuit)
        op_mv = self._next_opening_move(board, rows, cols, score_cols)
        if op_mv is not None:
            return op_mv

        # Record current board hash for repetition guard update
        try:
            current_hash = self._board_hash(board, rows, cols)
        except Exception:
            current_hash = None

        best_move = None
        max_depth = max(1, int(self.SEARCH_DEPTH))

        for depth in range(1, max_depth+1):
            if time.perf_counter() > deadline: break
            try:
                score, move = self._alphabeta(board, depth, float('-inf'), float('inf'),
                                              self.player, rows, cols, score_cols, start, deadline)
                if move is not None:
                    best_move = move
            except Exception:
                break

        if best_move is not None:
            # Final anti-cycle replacement: if chosen move would return to recent state, try next best
            if (self.prev_self_hashes or self.prev_after_hashes):
                ok, undo = self._apply_inplace(board, best_move, self.player, rows, cols, score_cols)
                if ok:
                    new_hash = self._board_hash(board, rows, cols)
                    self._undo_inplace(board, undo)
                    if (new_hash in self.prev_self_hashes) or (new_hash in self.prev_after_hashes):
                        # try alternative moves in order
                        alts = self._ordered_moves(board, self.player, rows, cols, score_cols)
                        for mv in alts:
                            ok2, un2 = self._apply_inplace(board, mv, self.player, rows, cols, score_cols)
                            if not ok2:
                                continue
                            nh2 = self._board_hash(board, rows, cols)
                            self._undo_inplace(board, un2)
                            if (nh2 not in self.prev_self_hashes) and (nh2 not in self.prev_after_hashes):
                                best_move = mv
                                break
            if current_hash is not None:
                self.prev_self_hashes.append(current_hash)
            # Update after-hash, last move and recent mutation squares
            try:
                ok3, un3 = self._apply_inplace(board, best_move, self.player, rows, cols, score_cols)
                if ok3:
                    after_hash = self._board_hash(board, rows, cols)
                    self.prev_after_hashes.append(after_hash)
                    self._undo_inplace(board, un3)
            except Exception:
                pass
            self.last_my_move = best_move
            if best_move.get("action") in ("flip","rotate"):
                fr = best_move.get("from");
                if fr:
                    self.recent_mutations.append((best_move["action"], int(fr[0]), int(fr[1])))
            return best_move

        # fallback: first legal move
        for mv in self._enumerate_moves(board, self.player, rows, cols, score_cols):
            ok, undo = self._apply_inplace(board, mv, self.player, rows, cols, score_cols)
            if ok:
                self._undo_inplace(board, undo)
                return mv
        if current_hash is not None:
            self.prev_self_hashes.append(current_hash)
        return None
