from typing import List, Tuple
import logging

import numpy as np


from tetris_app import TetrisApp, rotate_clockwise, check_collision

class TetrisAgent():
    def __init__(self, tetrisApp: TetrisApp):
        self.tetrisApp = tetrisApp


    def get_height_difference(self, board: np.ndarray) -> int:
        """ Returns sum of absolute height differences between each column """
        heights = (board != 0).argmax(axis=0)
        return np.sum(np.abs(np.diff(heights)))


    def get_total_height(self, board: np.ndarray) -> int:
        """ Returns total sum of column heights """
        heights = (board != 0).argmax(axis=0)
        return np.sum(board.shape[0] - 1 - heights)


    def get_number_of_holes(self, board: np.ndarray) -> int:
        """ Hole is defined as empty cell lower than column height """
        heights = (board != 0).argmax(axis=0)
        empty = (board == 0)
        sum = 0
        for column in range(board.shape[1]):
            sum += np.sum(empty[heights[column]:, column], axis=0)
        return sum


    def get_board_parameters(self, board) -> List[int]:
        return [
            self.get_height_difference(board),
            self.get_total_height(board),
            self.get_number_of_holes(board),
        ]


    def simulate_move(self, column: int, rotation_cnt: int) -> np.ndarray:
        stone = self.state['stone']
        if check_collision(self.state['board'], stone, (column, 0)):
            return None # unable to rotate stone

        for i in range(rotation_cnt):
            stone = rotate_clockwise(stone)
            if check_collision(self.state['board'], stone, (column, 0)):
               return None # unable to rotate stone

        stone = np.array(stone)
        board = np.copy(self.state['board'])
        heights = (board != 0).argmax(axis=0)
        min_height = np.min(heights[column:column+stone.shape[1]])

        height_offset = stone.shape[0]
        start_height = max(min_height - height_offset, 0)

        while not check_collision(board, stone, (column, start_height+1)):
            start_height += 1

        for i in range(stone.shape[0]):
            for j in range(stone.shape[1]):
                if stone[i][j]:
                    board[start_height + i][column + j] = stone[i][j]

        return board


    def clean_board(self, board: np.ndarray) -> int:
        """Modifies the board by removing full rows and returns number of cleared rows"""
        non_zero_rows = ((board != 0).sum(1) != board.shape[1])
        non_zero_rows[-1] = True
        new_board = board[non_zero_rows]
        
        rows_cleared = board.shape[0] - new_board.shape[0]
        empty_rows = np.zeros((rows_cleared, board.shape[1]), dtype=int)

        if rows_cleared > 0:
            board = np.concatenate((empty_rows, new_board), axis=0)
        return rows_cleared


    def evaluate_move(self, column: int, rotation_cnt: int) -> float:
        if (board := self.simulate_move(column, rotation_cnt)) is None:
            return 0

        rows_cleared = self.clean_board(board)
        if rows_cleared > 0:
            parameters = self.get_board_parameters(board)
            logging.info(rows_cleared, parameters)

        return 0

    def find_optimal_move(self) -> Tuple[int, int]:
        min_value = 1e9
        move = (-1, 0)
        for column in range(self.state['board'].shape[1]):
            for rotation_cnt in range(4):
                value = self.evaluate_move(column, rotation_cnt)
                if value < min_value:
                    min_value = value
                    move = (column, rotation_cnt)

        return move
        

    def start(self):
        self.tetrisApp.init()

        while(1):
            #sleep(1)            
            self.state = self.tetrisApp.get_state()
            optimal_move = self.find_optimal_move()

            """
            if not state["gameover"] and not self.tetrisApp.actions:
                print(state)
                print(self.tetrisApp.actions)
                actions = []
                if randint(1, 2) % 2:
                    for i in range (randint(1, 6)):
                        actions.append('LEFT')
                    actions.append('DOWN')
                else:
                    if randint(1, 2) % 2:
                        for i in range (randint(1, 6)):
                            actions.append('RIGHT')
                        actions.append('DOWN')

                self.tetrisApp.add_actions(actions)
            """
            self.tetrisApp.tick()
            

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = TetrisApp()
    agent = TetrisAgent(app)
    agent.start()
