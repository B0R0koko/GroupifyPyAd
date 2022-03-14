from Groupify.prefs import Preferences
from Groupify.algos.exclusion import ExclusionAlgo


if __name__ == "__main__":

    prefs_manager = Preferences(100, 20)
    prefs_manager.set_seed(420)
    prefs = prefs_manager.gen_discrete_preferences()
    ads = prefs_manager.gen_discrete_preferences()

    algo = ExclusionAlgo(prefs, ads)
    algo()
