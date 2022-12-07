from tetris_app import TetrisApp
from random import randint

class TetrisAgent():
    def __init__(self, tetrisApp: TetrisApp):
        self.tetrisApp = tetrisApp

    def start(self):
        self.tetrisApp.init()

        while(1):
            #sleep(1)            
            state = self.tetrisApp.get_state()
            
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
            self.tetrisApp.tick()
            

if __name__ == '__main__':
    app = TetrisApp()
    agent = TetrisAgent(app)
    agent.start()
