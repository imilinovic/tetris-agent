from typing import List, Tuple

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


    def get_parameters(self, state) -> List[int]:
        return [
            self.get_height_difference(state["board"]),
            self.get_total_height(state["board"]),
            self.get_number_of_holes(state["board"]),
        ]


    def simulate_move(self, column: int, rotation_cnt: int) -> np.ndarray:
        stone = self.state['stone']
        for i in range(rotation_cnt):
            stone = rotate_clockwise(stone)
            if check_collision(self.state['board'], stone, (0, column)):
               return None # unable to rotate stone

        stone = np.array(stone)
        board = np.copy(self.state['board'])
        heights = (board != 0).argmax(axis=0)
        min_height = np.min(heights[column:column+stone.shape[1]])

        height_offset = stone.shape[0]
        start_height = max(min_height - height_offset, 0)

        while not check_collision(board, stone, (start_height, column)):
            start_height += 1

        for i in range(stone.shape[0]):
            for j in range(stone.shape[1]):
                if stone[i][j]:
                    board[start_height + i][column + j] = stone[i][j]

        return board


    def evaluate_move(self, column: int, rotation_cnt: int) -> float:
        board = self.simulate_move(column, rotation_cnt)
        # TO DO
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
        return move
        

    def start(self):
        self.tetrisApp.init()

        while(1):
            #sleep(1)            
            self.state = self.tetrisApp.get_state()
            print(self.state)
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
    app = TetrisApp()
    agent = TetrisAgent(app)
    agent.start()
