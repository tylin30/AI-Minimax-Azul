
from advance_model import *
from utils import *
from math import inf
import numpy as np
import time
import random
from copy import deepcopy
import sys
import datetime

class myPlayer(AdvancePlayer):
    def __init__(self,_id):
        super().__init__(_id)
        self.num_moves = 0
        self.minimax_agent = None
        
    def SelectMove(self,moves,game_state):
        # figure out the tineout
        self.minimax_agent = minimax(moves,game_state,self.id)
        return self.minimax_agent.iterative_ab(moves,game_state)


class minimax(object):
    def __init__(self,moves,game_state,_player_id):
        self._root = minimax_node(None,game_state,moves,_player_id)
        self.width = 5

        self.player_id = _player_id
        self.opponent_id = game_state.players[1].id if game_state.players[0].id == _player_id else game_state.players[0].id
        self.num_nodes = 0
        self._time = -1
        self._timeout = 1000

        self.alpha = -inf
        self.beta = inf
        self.depth = inf
        self.minimax_val = -inf


    def iterative_ab(self,moves,game_state):
        depth = 2
        global max_num_move_expand
        if len(moves) >= 13:
            self.width = 13
            depth = 3
        elif len(moves) > 0 and len(moves) < 13:
            self.width = 5
            depth = 4
        return self.alpha_beta(moves,depth, game_state)
    
    def get_top_moves(self, player_id, moves, game_state):
        prioritised_moves = []
        for move in moves:
            self_award = self.move_score(player_id, move, game_state)
            prioritised_moves += [(self_award, move)]

        prioritised_moves.sort(key=lambda prioritised: prioritised[0], reverse=True)
        top_moves = [x[1] for x in prioritised_moves[:self.width]]

        return top_moves

    def alpha_beta(self,moves,depth, game_state):
        best_move = None
        value = -inf
        alpha = -inf
        beta = inf
        
        if len(moves) > self.width:
            moves = self.get_top_moves(self.player_id, moves, game_state)

        for m in moves:
            node = self.expand(self._root,m)
            min_eval = self.min_value(node,depth-1,alpha,beta)
            # get the maximum rewards of the minimiser
            if min_eval > value:
                best_move = m
                value = min_eval
            
            if value >= beta:
                self.minimax_val = value
                return best_move
            
            alpha = max(alpha,value)
        self.minimax_val = value

        return best_move
        
    def min_value(self,node, depth, alpha, beta):
        value = inf
        cutoff_res =  self.cutoff(node,depth)
        if cutoff_res is not None: return cutoff_res

        moves = node.available_moves
        if len(moves) > self.width:
            moves = self.get_top_moves(node._pid, moves, node._game_state)
        
        for m in moves:
            child = self.expand(node,m)

            value = min(value,self.max_value(child,depth-1,alpha,beta))

            if value <= alpha:
                return value
            beta = min(beta,value)
        return value

    def max_value(self,node,depth,alpha,beta):
        value = -inf
        cutoff_res =  self.cutoff(node,depth)
        if cutoff_res is not None: return cutoff_res

        moves = node.available_moves
        if len(moves) > self.width:
            moves = self.get_top_moves(node._pid, moves, node._game_state)
        
        for m in moves:
            child = self.expand(node,m)

            # if the child is the opponent we want to call minimise,
            # but if the child is us we want to call maximise
            value = max(value,self.min_value(child,depth-1,alpha,beta))

            if value >= beta:
                return value
            alpha = min(alpha,value)

        return value
    
    '''
    def get_top_moves(self, player_id, moves, game_state):
        prioritised_moves = []
        for move in moves:
            award = self.utility(player_id, move, game_state)
            prioritised_moves += [(award, move)]
        
        prioritised_moves.sort(key=lambda prioritised: prioritised[0], reverse=True)
        top_moves = [x[1] for x in prioritised_moves[:max_num_move_expand]]
        # return top n moves
        return top_moves
    '''

    def cutoff(self,node,depth):
        if node._is_terminal:
            return 10000*(node._player_scores[self.player_id] - node._player_scores[self.opponent_id])
        if node._end_of_round:
            #return self.evaluate_round_end(node)
            return self.evaluate_state(node)
        if depth == 0:
            return self.evaluate_state(node)
        return None

    '''
    This should use the same / similar score function as the greedy agent without deepcopying the state
    '''
    def evaluate_state(self,node):
        return self.utility(self.player_id,node._move,node._game_state) - 0.5*self.utility(self.opponent_id,node._move, node._game_state)

    # same as hillCliming in greedy
    def utility(self,pid, move, game_state):
        h_value = 0
        player = game_state.players[pid]
        grid = player.grid_state
        h_value += self.score_grid(player,grid)
        return h_value
    
    # same as hillCliming in greedy
    def move_score(self,pid, move, game_state):
        # where PID is the player who mde the move, move
        # given a node lets look at the moves
    
        h_value = 0
        _, _, tg = move
        # the player whose move we are analysing
        
        player = game_state.players[pid]
        
        h_value += tg.number * 0.5
        
        # if the dest is to floor line - simulate a floorline placement
        if tg.num_to_floor_line > 0:
            num_to_place = tg.num_to_floor_line
            for i in range(len(player.floor)):
                if player.floor[i]== 0 and num_to_place > 0:
                    h_value += tg.number*player.FLOOR_SCORES[i]
                    num_to_place-=1
        
        # get the grid state
        grid = player.grid_state

        if tg.num_to_pattern_line > 0:
            # now we need to look at the move
            dest = tg.pattern_line_dest
            

            # make sure the before value of the grid state is zero
            before_val = grid[dest][int(player.grid_scheme[dest][tg.tile_type])]

            # if we are making a move that corresponds to a full pattern line
            if player.lines_number[dest] + tg.num_to_pattern_line == dest+1:
                assert(before_val == 0)
                # this line will then be full
                grid[dest][int(player.grid_scheme[dest][tg.tile_type])] = 1
            
            # now we can score
            score = self.score_grid(player,grid)

            # revert the value of the grid to the old value
            grid[dest][int(player.grid_scheme[dest][tg.tile_type])] = before_val
            
            h_value += score
        else:
            h_value += self.score_grid(player,grid)
        return h_value

    def score_grid(self,player,grid):
        score_inc = 0

        # 1. Move tiles across from pattern lines to the wall grid

        for i in range(player.GRID_SIZE):
            # Is the pattern line full? If not it persists in its current
            # state into the next round.
            if player.lines_number[i] == i+1:
                tc = player.lines_tile[i]
                col = int(player.grid_scheme[i][tc])

                # Tile will be placed at position (i,col) in grid
                grid[i][col] = 1

                # count the number of tiles in a continguous line
                # above, below, to the left and right of the placed tile.
                above = 0
                for j in range(col-1, -1, -1):
                    val = grid[i][j]
                    above += val
                    if val == 0:
                        break
                below = 0
                for j in range(col+1,player.GRID_SIZE,1):
                    val = grid[i][j]
                    below +=  val
                    if val == 0:
                        break
                left = 0
                for j in range(i-1, -1, -1):
                    val = grid[j][col]
                    left += val
                    if val == 0:
                        break
                right = 0
                for j in range(i+1, player.GRID_SIZE, 1):
                    val = grid[j][col]
                    right += val
                    if val == 0:
                        break

                # If the tile sits in a contiguous vertical line of
                # tiles in the grid, it is worth 1*the number of tiles
                # in this line (including itself).
                if above > 0 or below > 0:
                    score_inc += (1 + above + below)

                # In addition to the vertical score, the tile is worth
                # an additional H points where H is the length of the
                # horizontal contiguous line in which it sits.
                if left > 0 or right > 0:
                    score_inc += (1 + left + right)

                # If the tile is not next to any already placed tiles
                # on the grid, it is worth 1 point.
                if above == 0 and below == 0 and left == 0 \
                    and right == 0:
                    score_inc += 1

        # Score penalties for tiles in floor line
        penalties = 0
        for i in range(len(player.floor)):
            penalties += player.floor[i]*player.FLOOR_SCORES[i]
            
        # Players cannot be assigned a negative score in any round.
        score_change = score_inc + penalties
        if score_change < 0 and player.score < -score_change:
            score_change = -player.score
        
        total_score = player.score + score_change
        return total_score

    # create a new node based off the old node and the move to play
    def expand(self,node,move):
        self.num_nodes+=1
        # player to move
        pid = node._pid
        gs = deepcopy(node._game_state)
        player_order = []
        for i in range(gs.first_player, 2):
            player_order.append(i)

        for i in range(0, gs.first_player):
            player_order.append(i)

        # if this is the first player in the player order to make a
        # move, we need to return the node and then
        if player_order[0] == pid:
            pid = player_order[0]
        else:
            pid = player_order[1]
        game_continuing = node._game_continuing

        if game_continuing:
            plr_state = gs.players[pid]
            gs.ExecuteMove(pid, move)

            # if there are no tiles remaining after the player has made a move, we then need
            # to either end the round and continue with a new round or end the game
            if not gs.TilesRemaining():
                gs.ExecuteEndOfRound()

                # Is it the end of the game?
                for i in player_order:
                    plr_state = gs.players[i]
                    completed_rows = plr_state.GetCompletedRows()

                    if completed_rows > 0:
                        # this is the end of the game
                        player_scores = {}
                        for p in gs.players:
                            p.EndOfGameScore()
                            player_scores[p.id] = plr_state.score

                        new_node = minimax_node(move,gs,[],None,player_scores=player_scores)
                        new_node._is_terminal = True
                        new_node._game_continuing = False
                        new_node._parent = node
                        # i dont know if this is right
                        node.add_child(new_node)
                        return new_node

                # Set up the next round
                if game_continuing:
                    player_scores = {}
                    for p in gs.players:
                        player_scores[p.id] = p.score

                    new_node = minimax_node(move,gs,[],None,player_scores=player_scores)
                    new_node._parent = node
                    new_node._game_continuing = True
                    new_node._is_terminal = False
                    new_node._end_of_round = True
                    node.add_child(new_node)
                    return new_node
            else:
                # we can return the current node
                # get the player state of the next player
                next_player_id = player_order[1] if pid == player_order[0] else player_order[0]
                next_player = gs.players[next_player_id]
                assert(next_player_id == abs(node._pid -1))
            # get available moves
                new_moves = next_player.GetAvailableMoves(gs)
                new_node = minimax_node(move,gs,new_moves,next_player.id)
                new_node._parent = node
                new_node._game_continuing = True
                new_node._is_terminal = False
                node.add_child(new_node)
                return new_node
        else:
            return None


class minimax_node(object):
    # we don't need player information
    def __init__(self,move,game_state,moves,player_id,player_scores=None):

        # make sure to copy moves game state and move

        # the move to get to this game state
        self._move = move

        # current game state of the board
        self._game_state = game_state

        # player to make a move
        self._pid = player_id

        # replace this by a call to model to find the next moves
        self.available_moves = moves

        self._children = []

        # statistics used to calculate ucb policy
        self._parent = None
        self._end_of_round = False
        self._game_continuing = True
        self._is_terminal = False
        self._player_scores = player_scores
        # self._is_start = True

    def add_child(self,node):
        self._children.append(node)
    


