#
#
#
# /$$$$$$$$                     /$$                     /$$                      /$$$$$$  /$$
# | $$_____/                    | $$                    |__/                     /$$__  $$| $$
# | $$       /$$   /$$  /$$$$$$$| $$ /$$   /$$  /$$$$$$$ /$$  /$$$$$$  /$$$$$$$ | $$  \ $$| $$  /$$$$$$   /$$$$$$
# | $$$$$   |  $$ /$$/ /$$_____/| $$| $$  | $$ /$$_____/| $$ /$$__  $$| $$__  $$| $$$$$$$$| $$ /$$__  $$ /$$__  $$
# | $$__/    \  $$$$/ | $$      | $$| $$  | $$|  $$$$$$ | $$| $$  \ $$| $$  \ $$| $$__  $$| $$| $$  \ $$| $$  \ $$
# | $$        >$$  $$ | $$      | $$| $$  | $$ \____  $$| $$| $$  | $$| $$  | $$| $$  | $$| $$| $$  | $$| $$  | $$
# | $$$$$$$$ /$$/\  $$|  $$$$$$$| $$|  $$$$$$/ /$$$$$$$/| $$|  $$$$$$/| $$  | $$| $$  | $$| $$|  $$$$$$$|  $$$$$$/
# |________/|__/  \__/ \_______/|__/ \______/ |_______/ |__/ \______/ |__/  |__/|__/  |__/|__/ \____  $$ \______/
#                                                                                              /$$  \ $$
#                                                                                             |  $$$$$$/
#                                                                                              \______/
#
#
#
# --Mironov-Mikhail-BEC-191-----------------------------------------------------------------------------------------------------------

import torch
from collections import deque


class ExclusionAlgo:
    def __init__(self, prefs: torch.tensor, ads: torch.tensor):
        self.prefs = prefs
        self.ads = ads
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._target_indecies = deque(range(len(self.prefs)))
        self._excluded_indecies = deque([])
        # Get dot products of all ads onto themselves
        self._dot_array = torch.einsum("ij,ij->i", self.ads, self.ads)

    # Function for calculation of overall rho-scores at each exclusion
    def _calc_overall_score(
        self, target_weight: float, target_indecies: deque, excluded_indecies: deque
    ) -> torch.float:
        # Calculate scores of satisfactions both for target and exclusion sets
        target_score = torch.mean(
            torch.matmul(self.prefs[target_indecies], self.ads.T) / self._dot_array,
            axis=0,
        )
        excluded_score = torch.mean(
            torch.matmul(self.prefs[excluded_indecies], self.ads.T) / self._dot_array,
            axis=0,
        )
        return target_weight * target_score + (1 - target_weight) * excluded_score

    # Find best person to remove and best corresponding ad in terms of rho-score
    def _find_best_exclusion(self, target_indecies: deque, excluded_indecies: deque):
        # If we excluded a player what the best ad yields in terms of score
        scores_at_exclusion = torch.zeros(
            len(target_indecies), dtype=torch.float16, device=self.device
        )
        best_ads_at_exclusion = torch.zeros(len(target_indecies), device=self.device)
        # Copy temporary indecies not to mess up with global indexation of players
        tmp_target_indecies, tmp_excluded_indecies = (
            target_indecies.copy(),
            excluded_indecies.copy(),
        )
        # Iterate through each potential exclusion among those who are left in target set
        for i in range(len(target_indecies)):
            # Temporarily exclude player from target set and move him to exclusion set
            excluded_player = tmp_target_indecies.popleft()
            tmp_excluded_indecies.append(excluded_player)
            # Now find best fit ad with such groups
            overall_scores = self._calc_overall_score(
                0.5, tmp_target_indecies, tmp_excluded_indecies
            )
            best_ad_at_exclusion_indecies = torch.argsort(
                overall_scores, descending=True
            )  # --> scores are sorted and corresponding indecies are returned
            best_ad_at_exclusion_idx = best_ad_at_exclusion_indecies[0]
            best_ad_at_exclusion_score = overall_scores[best_ad_at_exclusion_idx]
            # Revert changes in tmp deques
            tmp_target_indecies.append(excluded_player)
            tmp_excluded_indecies.pop()
            # Save overall_score and index of best advertisement at current exclusion
            scores_at_exclusion[i] = best_ad_at_exclusion_score
            best_ads_at_exclusion[i] = best_ad_at_exclusion_idx

        # Find best person to exclude and therefore best corresponding advertisement
        # best person to exclude at i-th exclusion (relative index)
        rel_best_person_to_exclude_idx = torch.argsort(
            scores_at_exclusion, descending=True
        )[0]
        # best person to exclude at i-th exclusion (global index)
        global_best_person_to_exclude_idx = target_indecies[
            rel_best_person_to_exclude_idx
        ]
        # If the best person is excluded - best rho-score
        best_exclusion_score = scores_at_exclusion[rel_best_person_to_exclude_idx]
        # Best advertisement in terms of rho-score at i-th exclusion
        best_ad_at_exclusion = best_ads_at_exclusion[rel_best_person_to_exclude_idx]
        return (
            global_best_person_to_exclude_idx,
            best_exclusion_score.item(),
            best_ad_at_exclusion.item(),
        )

    def __call__(self):

        self.best_exclusion_outputs = []
        target_indecies, excluded_indecies = (
            self._target_indecies.copy(),
            self._excluded_indecies.copy(),
        )
        n_iters = len(self._target_indecies) - 1

        for _ in range(n_iters):
            best_exclusion_output = self._find_best_exclusion(
                target_indecies, excluded_indecies
            )
            print(best_exclusion_output)
            self.best_exclusion_outputs.append(best_exclusion_output)
            best_person_to_exclude_idx = best_exclusion_output[0]
            # We can safely remove indecies since there is no duplicates. Remove best exclusion from target set and
            # assign such person to excluded set
            target_indecies.remove(best_person_to_exclude_idx)
            excluded_indecies.append(best_person_to_exclude_idx)
