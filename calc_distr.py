import random


class CalcDistr():

    def __init__(self, real_inferred, local_radius = 0.0025, lower_bound = 0, 
                 upper_bound = 1, uniform_chance = 0.05):
        self.local_radius = local_radius
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.uniform_chance = uniform_chance
        self.real = real_inferred[0]
        self.inferred = real_inferred[1]
        self.losses = [(r - i) ** 2 for r, i in zip(self.real, self.inferred)]
        self.max_loss = max(self.losses)

    def __call__(self):

        if random.uniform(0, 1) < self.uniform_chance:
            gen_val = random.uniform(self.lower_bound, self.upper_bound)
        else:
            gen_accepted = False
            while not gen_accepted:
                r_idx = random.randint(0, len(self.real) - 1)
                if self.losses[r_idx] >= random.uniform(0, self.max_loss):
                    gen_accepted = True
                    gen_val = random.uniform(
                        max(0, self.real[r_idx] - self.local_radius),
                        min(1, self.real[r_idx] + self.local_radius))

        return gen_val


