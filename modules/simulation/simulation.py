import sys
from threading import Thread


class Simulation:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.rendering = True
        self.state = "playing"
        self.action = None
        thread = Thread(target=self.process_user_input, daemon=True)
        thread.start()

    def run(self):
        while self.state != "done":
            while self.state == "paused":
                continue
            state, reward = self.env.step(self.action)
            self.action = self.agent.step(state, reward)
            if self.rendering:
                self.env.render(self.agent)

    def process_user_input(self):
        self.print_help_message()
        user_input = ""
        while user_input != "exit":
            try:
                user_input = input(">:")
                command_type = self.get_argument(user_input, 0)
                commands = {
                    "exit": self.exit_program,
                    "render": self.render, "r": self.render,
                    "reset": self.reset,
                    "play": self.play,
                    "pause": self.pause,
                    "predict": self.predict, "p": self.predict,
                    "train": self.train_speed, "t": self.train_speed,
                    "save": self.save, "s": self.save,
                    "dream": self.dream, "d": self.dream,
                    "view": self.view, "v": self.view,
                }
                command = commands.get(command_type, lambda x: self.invalid_command())
                command(user_input)
            except:
                self.invalid_command()

    def get_num_args(self, command):
        return len(command.split())

    def get_argument(self, command, arg_num):
        arguments = command.split()
        if arg_num >= len(arguments):
            raise Exception(f"Expected an argument at position {arg_num} but found None")
        argument = arguments[arg_num].replace(" ", "")
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
        print("Playing simulation")
        return

    def pause(self, command):
        self.state = "paused"
        print("Pausing simulation")
        return

    def train_speed(self, command):
        try:
            train_interval = int(self.get_argument(command, 1))
            if train_interval == 0:
                train_interval = -1
            self.agent.train_interval = train_interval
            print(f"Set train interval to {train_interval}")
        except Exception as e:
            print(e)
        return

    def predict(self, command):
        self.agent.predicting = not self.agent.predicting
        print(f"Prediction {'enabled' if self.agent.predicting else 'disabled'}")
        return

    def dream(self, command):
        self.agent.dreaming = not self.agent.dreaming
        print(f"Dreaming {'enabled' if self.agent.dreaming else 'disabled'}")
        return

    def view(self, command):
        try:
            view = int(self.get_argument(command, 1))
        except Exception as e:
            view = 1 - min(self.env.view, 1)
        if self.agent.visualizer is not None:
            self.agent.visualizer.active = view != 0
        self.env.view = view
        print(f"Viewmode: {self.env.view}")
        return

    def save(self, command):
        try:
            save_interval = int(self.get_argument(command, 1))
            if save_interval == 0:
                save_interval = -1
            self.agent.save_interval = save_interval
            print(f"Set save interval to {save_interval}")
        except Exception as e:
            self.agent.save_model()
            print(f"Quicksaving {self.agent.model_name()}")
        return

    def reset(self, command):
        self.env.reset()
        print("Environment reset")
        return

    def invalid_command(self):
        print("Error: Invalid command")
