class SigmoidSlopeScheduler:
    "Exponential decay."

    def __init__(self, model, start, end, total_steps):
        self.model = model
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.step_idx = 0

    def step(self):
        t = min(self.step_idx / self.total_steps, 1.0)
        value = self.start * (self.end / self.start) ** t
        self.model.sigmoid_slope.fill_(value)
        self.step_idx += 1
