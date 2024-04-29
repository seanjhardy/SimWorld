from gym.envs.classic_control import rendering


class Wall:
    colour = (100, 100, 100, 1)

    @staticmethod
    def render(viewer, x, y, size):
        t = rendering.Transform(translation=(x * size, y * size))
        geom = viewer.draw_polygon([[0, 0], [0, size], [size, size], [size, 0]], filled=True, color=Wall.colour)
        geom.add_attr(t)
