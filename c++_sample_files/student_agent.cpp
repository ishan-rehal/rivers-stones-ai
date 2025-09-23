#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <random>

namespace py = pybind11;


/*
=========================================================
 STUDENT AGENT FOR STONES & RIVERS GAME
---------------------------------------------------------
 The Python game engine passes the BOARD state into C++.
 Each board cell is represented as a dictionary in Python:

    {
        "owner": "circle" | "square",          // which player owns this piece
        "side": "stone" | "river",             // piece type
        "orientation": "horizontal" | "vertical"  // only relevant if side == "river"
    }

 In C++ with pybind11, this becomes:

    std::vector<std::vector<std::map<std::string, std::string>>>

 Meaning:
   - board[y][x] gives the cell at (x, y).
   - board[y][x].empty() → true if the cell is empty (no piece).
   - board[y][x].at("owner") → "circle" or "square".
   - board[y][x].at("side") → "stone" or "river".
   - board[y][x].at("orientation") → "horizontal" or "vertical".

=========================================================
*/

// ---- Move struct ----
struct Move {
    std::string action;
    std::vector<int> from;
    std::vector<int> to;
    std::vector<int> pushed_to;
    std::string orientation;
};

// ---- Student Agent ----
class StudentAgent {
public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()) {}

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board, int row, int col, const std::vector<int>& score_cols, float current_player_time, float opponent_time) {
        int rows = board.size();
        int cols = board[0].size();

        std::vector<Move> moves;

        // Directions
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        // Iterate over board
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                const auto &cell = board[y][x];
                if (cell.empty()) continue;

                if (cell.at("owner") != side) continue; // only my pieces

                std::string side_type = cell.at("side");

                // ---- MOVES ----
                for (auto [dx,dy] : dirs) {
                    int nx = x+dx, ny = y+dy;
                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;

                    if (board[ny][nx].empty()) {
                        moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
                    }
                }

                // ---- PUSHES ----
                for (auto [dx,dy] : dirs) {
                    int nx = x+dx, ny = y+dy;
                    int nx2 = x+2*dx, ny2 = y+2*dy;
                    if (nx<0||ny<0||nx>=cols||ny>=rows) continue;
                    if (nx2<0||ny2<0||nx2>=cols||ny2>=rows) continue;

                    if (!board[ny][nx].empty() && board[ny][nx].at("owner") != side
                        && board[ny2][nx2].empty()) {
                        moves.push_back({"push", {x,y}, {nx,ny}, {nx2,ny2}, ""});
                    }
                }

                // ---- FLIP ----
                if (side_type == "stone") {
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "horizontal"});
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "vertical"});
                }

                // ---- ROTATE ----
                if (side_type == "river") {
                    moves.push_back({"rotate", {x,y}, {x,y}, {}, ""});
                }
            }
        }

        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""}; // fallback
        }

        std::uniform_int_distribution<> dist(0, moves.size()-1);
        return moves[dist(gen)];
    }

private:
    std::string side;
    std::random_device rd;
    std::mt19937 gen;
};

// ---- PyBind11 bindings ----
PYBIND11_MODULE(student_agent_module, m) {
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose);
}