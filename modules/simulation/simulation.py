import sys
from threading import Thread

from modules.controller.agentController import AgentController


class Simulation:
    def __init__(self, env):
        self.env = env
        self.rendering = True
        self.state = "playing"
        thread = Thread(target=self.process_user_input, daemon=True)
        thread.start()

    def run(self):
        while self.state != "done" and self.state != "paused":
            self.env.simulate()
            if self.rendering:
                self.env.render()

    def process_user_input(self):
        self.print_help_message()
        userInput = ""
        while userInput != "exit":
            try:
                userInput = input(">:")
                commandType = self.get_argument(userInput, 0)
                commands = {
                    "exit": self.exit_program,
                    "render": self.render,
                    "reset": self.reset,
                    "train": self.toggle_train,
                    "play": self.play,
                    "pause": self.pause,
                    "predict": self.predict,
                    "train_speed": self.train_speed,
                    "save": self.save,
                }
                command = commands.get(commandType, lambda x: self.invalid_command())
                command(userInput)
            except:
                self.invalid_command()

    def get_num_args(self, command):
        return len(command.split())

    def get_argument(self, command, argNum):
        arguments = command.split()
        if argNum >= len(arguments):
            return "INVALID"
        argument = arguments[argNum].replace(" ", "")
        return argument

    def print_help_message(self):
        print("===============")
        print("Running:", self.env.name)
        print("===============")

    def exit_program(self, command):
        print("Shutting down env")
        self.state = "done"
        sys.exit()

    def render(self, command):
        self.rendering = not self.rendering
        print(f"Rendering {'enabled' if self.rendering else 'disabled'}")
        return

    def play(self, command):
        self.state = "playing"
        self.run()
        print("Playing simulation")
        return

    def pause(self, command):
        self.state = "paused"
        print("Pausing simulation")
        return

    def toggle_train(self, command):
        AgentController.training = not AgentController.training
        print(f"Training {'enabled' if AgentController.training else 'disabled'}")
        return

    def train_speed(self, command):
        train_speed = self.get_argument(command, 1)
        AgentController.train_speed = int(train_speed)
        print(f"Set train speed to {train_speed}")
        return

    def predict(self, command):
        AgentController.predicting = not AgentController.predicting
        print(f"Prediction {'enabled' if AgentController.predicting else 'disabled'}")
        return

    def save(self, command):
        AgentController.save = not AgentController.save
        print(f"Saving {'enabled' if AgentController.save else 'disabled'}")
        return

    def reset(self, command):
        self.env.reset()
        print("Environment reset")
        return

    def invalid_command(self):
        print("Error: Invalid command")
