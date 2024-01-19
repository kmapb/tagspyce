"""
Generative model for a power-law distributed investment, as
early stage venture seems empirically to be. See, e.g., Abe Othman's
2020 work at AngellList.

https://www.angellist.com/blog/what-angellist-data-says-about-power-law-returns-in-venture-capital

We take as our jumping off point Mitzenmacher's "A Brief History of Generative
Models for Power Law and Lognormal Distributions" (2003).

https://projecteuclid.org/journals/internet-mathematics/volume-1/issue-2/A-Brief-History-of-Generative-Models-for-Power-Law-and/im/1089229510.full
"""
from numpy.random import normal

class Company:
    def __init__(self,
                 name: str,
                 valuation: float,
                 age: float):
        self.name = name
        self.valuation = valuation
        self.age = age
        self.state = "operating"
        self.momentum = 0.0
        self.momentum_decay = 0.1

    def calculate_daily_improvement(self):
        # XXX: shouldn't larger companies find it harder to bleed off momentum? And
        # have a tighter day-to-day variance?
        new = normal(1.01, 0.01)
        alpha = self.momentum_decay
        self.momentum = (1.0 - alpha) * self.momentum + alpha * new
        return self.momentum

    def __str__(self):
        return "{}: {}".format(self.name, self.valuation)

    def age_days(self, num_days=1):
        # Change states with probability; mutate value; etc.
        for i in range(num_days):
            self.valuation *= self.calculate_daily_improvment()
            self.age += 1.0

class LateStagePortfolio(Portfolio):
    def __init__(self, name, valuation):
        StaticCompany.__init__(self, name, valuation)

    def days(self, num_days=1):
        sppendelf.valuation *= norm