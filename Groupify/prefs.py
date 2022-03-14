import torch


# Class used to mine and generate preferences
class Preferences:
    def __init__(self, n_players, n_topics):
        self.n_players = n_players
        self.n_topics = n_topics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_seed(self, seed_num):
        torch.manual_seed(seed_num)

    # Mine preferences from known searches using BertTopics
    def extract_keywords(self):
        return NotImplementedError

    # Generate discrete preferences from given distribution ("0" - doesnt like topic, "1"- likes topic)
    def gen_discrete_preferences(self):
        return torch.normal(
            mean=0.5,
            std=0.166,
            size=(self.n_players, self.n_topics),
            device=self.device,
        ).round()

    def __str__(self):
        return f"Preferences:\nPlayers: {self.n_players}\nTopics: {self.n_topics}"


if __name__ == "__main__":
    pref = Preferences(20, 10)
    print(pref.gen_discrete_preferences())
    print(pref)
