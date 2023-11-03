# Constants
default_skill_value = 0 # value for a players skill if no player_skill is provided on player creation
skill_value_modifier = 0.03
total_of_decks = 11 # total amount of decks
player_num = 512 # total number of players in a tournament
num_rounds = 14 # total number of rounds in a tournament



# Player Tournament and classes
import numpy as np
import pandas as pd
from threading import Lock
from itertools import combinations
import heapq

class Deck:
    def __init__(self, name):
        self.name = name
        self.matchup_spread = {}

    def set_matchup_win_prob(self, opponent_deck_name, win_prob):
        self.matchup_spread[opponent_deck_name] = win_prob

    def get_matchup_win_prob(self, opponent_deck_name):
        return self.matchup_spread.get(opponent_deck_name, 0.5)
    
    def __str__(self):
        return self.name
    
    def __reduce__(self):
        # Return the class itself, arguments, and the state
        return (self.__class__, (self.name,), {'name': self.name, 'matchup_spread': self.matchup_spread})
    
    def __setstate__(self, state):
        # Set the object's state from the given state dictionary
        self.__dict__.update(state)

class DeckManager:
    def __init__(self):
        self.decks = {}

    def add_deck(self, deck):
        self.decks[deck.name] = deck

    def generate_win_probabilities(self, deck_names):
        num_decks = len(deck_names)
        for i in range(num_decks):
            for j in range(i + 1, num_decks):
                win_prob = np.clip(np.random.normal(0.5, 0.15), 0, 1)
                self.decks[deck_names[i]].set_matchup_win_prob(deck_names[j], win_prob)
                self.decks[deck_names[j]].set_matchup_win_prob(deck_names[i], 1 - win_prob)

    def get_win_prob_matrix(self):
        deck_names = list(self.decks.keys())
        num_decks = len(deck_names)
        matrix = np.zeros((num_decks, num_decks))
        for i in range(num_decks):
            for j in range(num_decks):
                matrix[i, j] = self.decks[deck_names[i]].get_matchup_win_prob(deck_names[j])
        return matrix

    def get_win_prob_dataframe(self):
        deck_names = list(self.decks.keys())
        num_decks = len(deck_names)
        matrix = np.zeros((num_decks, num_decks))
        for i in range(num_decks):
            for j in range(num_decks):
                matrix[i, j] = self.decks[deck_names[i]].get_matchup_win_prob(deck_names[j])
        df = pd.DataFrame(matrix, index=deck_names, columns=deck_names)
        return df

    def load_win_probabilities_from_csv(self, file_path):
        # Read the Excel file
        df = pd.read_csv(file_path, header=0, index_col=0)
        
        # Drop the Representation column if it exists
        if 'Representation' in df.columns:
            df = df.drop(columns=['Representation'])
        
        # Ensure decks are added to the manager
        deck_names = df.index.to_list()
        for deck_name in deck_names:
            if deck_name not in self.decks:
                self.add_deck(Deck(deck_name))
        
        # Populate the matchup probabilities
        for i, row in df.iterrows():
            for j, value in row.items():
                self.decks[i].set_matchup_win_prob(j, value)
    
    def __reduce__(self):
        # Return the class itself, no arguments, and the state
        return (self.__class__, (), {'decks': self.decks})
    
    def __setstate__(self, state):
        # Set the object's state from the given state dictionary
        self.__dict__.update(state)

class Player:
    def __init__(self, player_id, alias, player_skill=default_skill_value, deck=None):
        self.id = player_id
        self.alias = alias
        self.skill = player_skill
        self.deck = deck
        self.wins = 0
        self.losses = 0
        self.history = []  # List to store match results
        self.history_lock = Lock()
    
    def __reduce__(self):
        # The object's state is returned as a tuple:
        # (callable, arguments_to_callable, additional_state)
        # Lock object is not pickled, so we're not including it in the state.
        return (self.__class__, (self.id, self.alias, self.skill, self.deck), {'wins': self.wins, 'losses': self.losses, 'history': self.history})
    
    def __setstate__(self, state):
        self.wins = state.get('wins', 0)
        self.losses = state.get('losses', 0)
        self.history = state.get('history', [])
        self.history_lock = Lock()  # Initialize a new Lock object after unpickling

    def set_deck(self, deck):
        """Set the deck for the player."""
        self.deck = deck

    def get_deck(self):
        """Retrieve the player's deck."""
        return self.deck

    def record_match(self, tournament, tournament_round, opponent, result):
        with self.history_lock:
            self.history.append([tournament.name, tournament_round, str(opponent.id) + opponent.alias, result])
    
    def get_tournament_vector(self):
        # Assuming result is the last element in the history entry
        return np.array([1 if entry[-1] == "W" else -1 for entry in self.history])
    
    def get_tournament_history(self, tournament):
        tournament_results = [history for history in self.history if history[0] == tournament.name]
        return tournament_results

    def get_tournament_standing(self, tournament):
        tournament_results = [history for history in self.history if history[0] == tournament.name]
        wins = 0
        losses = 0
        for match in tournament_results:
            result = match[-1]
            if result == "W":
                wins += 1
            elif result == "L":
                losses += 1
        return (wins, losses)

    def get_tournament_score(self, tournament, decay_factor=1):
        """
        Calculate the player's tournament score with a decay factor.
        Earlier wins are worth more than later wins.
        """
        tournament_results = self.get_tournament_history(tournament)
        score = 0
        base_score = len(tournament_results)  # Start scoring from the number of matches.
        
        # Loop over the results in reverse (to handle the decay factor correctly).
        for result in reversed(tournament_results):
            if result[-1] == "W":
                score += base_score ** decay_factor
            base_score -= 1  # Decrease the base score for the next round.
        
        return score

    def get_opponents_match_win_percentage(self, tournament):
        """
        Calculate the Opponents' Match-Win Percentage for the player.
        """
        opponents_win_percentage = []
        for match in self.get_tournament_history(tournament):
            opponent_id = match[2]  # Assuming the opponent's ID is the third item in the history.
            opponent = tournament.get_player_by_id(opponent_id)
            if opponent:
                wins, _ = opponent.get_tournament_standing(tournament)
                total_matches = len(opponent.get_tournament_history(tournament))
                if total_matches > 0:  # To avoid division by zero
                    opponents_win_percentage.append(wins / total_matches)
        
        # Calculate the average opponents' win percentage.
        return sum(opponents_win_percentage) / len(opponents_win_percentage) if opponents_win_percentage else 0

    def __str__(self):
        return f"[{self.id}] {self.alias} ({self.get_deck().name}) - Skill Level: {self.skill}"

    def __repr__(self):
        return f"[{self.id}] {self.alias} ({self.get_deck().name}) - Skill Level: {self.skill}"


class Tournament:
    def __init__(self, name, player_aliases, win_prob_matrix):
        self.name = name
        self.players = player_aliases
        self.win_prob_matrix = win_prob_matrix
        self.rounds_played = 0
        self.current_draft_standings = {player: [] for player in self.players}  # Use lists to hold match history

    def __reduce__(self):
        # Assuming win_prob_matrix is a numpy array, which is pickleable.
        # players list is also assumed to be pickleable.
        return (self.__class__, (self.name, self.players, self.win_prob_matrix), {'rounds_played': self.rounds_played})

    def __setstate__(self, state):
        self.rounds_played = state.get('rounds_played', 0)

    def calculate_similarity_score(self, vector1, vector2):
        """
        Calculate the similarity score between two tournament vectors.
        This uses the dot product to reward similarity in patterns of wins and losses.
        """
        # Normalize the vectors to unit vectors to only compare the patterns, not the lengths.
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        
        # Compute the dot product
        similarity = np.dot(unit_vector1, unit_vector2)
        
        return similarity

    def calculate_similarity_scores(self):
        # Initialize a dictionary to hold the sum of similarity scores for each player
        similarity_scores = {player: 0 for player in self.players}

        # Create a vector for each player
        player_vectors = {player: player.get_tournament_vector() for player in self.players}

        # Calculate the similarity score between each pair of players
        for i, player1 in enumerate(self.players):
            for player2 in self.players[i+1:]:  # Avoid repeating comparisons
                score = self.calculate_similarity_score(player_vectors[player1], player_vectors[player2])
                similarity_scores[player1] += score
                similarity_scores[player2] += score

        return similarity_scores

    def sort_by_similarity(self):
        # Calculate the similarity scores
        similarity_scores = self.calculate_similarity_scores()

        # Sort players by their similarity score
        sorted_players = sorted(self.players, key=lambda player: similarity_scores[player], reverse=True)

        return sorted_players

    def swiss_pairings(self):
        sorted_players = self.sort_by_similarity()
        
        pairings = []
        while sorted_players:
            p1 = sorted_players.pop(0)
            # Check if there's another player to pair with, if not, p1 gets a bye
            p2 = sorted_players.pop(0) if sorted_players else None
            pairings.append((p1, p2)) if p2 else pairings.append((p1, 'Bye'))
            
        return pairings

    def create_pods(self):
        # Create pods based on current standings
        sorted_players = sorted(self.players, key=lambda x: x.get_tournament_score(self), reverse=True)
        return [sorted_players[i:i+8] for i in range(0, len(sorted_players), 8)]

    def calculate_draft_similarity_score(self, history1, history2):
        # Assuming histories are strings like "WWLLW", "LWWLW" etc.
        score = sum(1 for h1, h2 in zip(history1, history2) if h1[-1] == h2[-1])
        return score

    def sort_players_by_draft_similarity(self, pod, standings):
        # Calculate similarity scores for each combination of players in the pod
        similarity_scores = {}
        for p1, p2 in combinations(pod, 2):
            score = self.calculate_draft_similarity_score(standings[p1], standings[p2])
            similarity_scores[(p1, p2)] = score
        
        # Sort the list of players based on similarity scores
        sorted_player_pairs = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_players_by_similarity = []
        while sorted_player_pairs:
            highest_pair = sorted_player_pairs.pop(0)[0]
            sorted_players_by_similarity.extend(highest_pair)
            # Remove pairs involving players already paired
            sorted_player_pairs = [(pair, score) for pair, score in sorted_player_pairs if highest_pair[0] not in pair and highest_pair[1] not in pair]
        
        # We might have duplicates since players can be part of multiple pairs, so we'll deduplicate them
        seen = set()
        sorted_players_by_similarity = [x for x in sorted_players_by_similarity if not (x in seen or seen.add(x))]
        return sorted_players_by_similarity

    def draft_swiss_pairings(self, pod):
        # Sort players by their win records first
        sorted_pod_players = sorted(pod, key=lambda x: self.current_draft_standings[x], reverse=True)
        # Then sort by similarity of their tournament history
        sorted_pod_players = self.sort_players_by_draft_similarity(sorted_pod_players, self.current_draft_standings)
        
        pairings = []
        while sorted_pod_players:
            p1 = sorted_pod_players.pop(0)
            # Since it's a draft pod, we're guaranteed to have an even number of players, so no need for a bye.
            p2 = sorted_pod_players.pop(0)
            pairings.append((p1, p2))
            
        return pairings

    def simulate_round(self):
        pairings = self.swiss_pairings()
        self.rounds_played += 1
        for p1, p2 in pairings:
            index_A = self.players.index(p1)
            index_B = self.players.index(p2)
            r = np.random.rand()
            if r < self.win_prob_matrix[index_A, index_B]:
                p1.record_match(self, self.rounds_played, p2, 'W')
                p2.record_match(self, self.rounds_played, p1, 'L')
            else:
                p1.record_match(self, self.rounds_played, p2, 'L')
                p2.record_match(self, self.rounds_played, p1, 'W')

    # round simulation
    def simulate_constructed_rounds(self, n_rounds):
        for _ in range(n_rounds):
            self.simulate_round()
    
    def simulate_draft_round(self, pod):
        pairings = self.draft_swiss_pairings(pod)
        self.rounds_played += 1
        for p1, p2 in pairings:
            p1_win_percentage = 0.5 + skill_value_modifier*(p1.skill - p2.skill)
            r = np.random.rand()
            if r < p1_win_percentage:
                p1.record_match(self, self.rounds_played, p2, 'W')
                p2.record_match(self, self.rounds_played, p1, 'L')
                self.current_draft_standings[p1].append('W')  # Append 'W' to the list for player 1
                self.current_draft_standings[p2].append('L')  # Append 'L' to the list for player 2
            else:
                p1.record_match(self, self.rounds_played, p2, 'L')
                p2.record_match(self, self.rounds_played, p1, 'W')
                self.current_draft_standings[p1].append('L')  # Append 'W' to the list for player 2
                self.current_draft_standings[p2].append('W')  # Append 'L' to the list for player 1
    
    def simulate_draft_rounds(self, n_rounds):
        self.current_draft_standings = {player: [] for player in self.players}  # Use lists to hold match history
        pods = self.create_pods()
        for _ in range(n_rounds):
            for pod in pods:
                self.simulate_draft_round(pod)
        

    def simulate_tournament(self):
        # Format for worlds (is it really 4 cc - 3 draft - 3 draft - 4 cc and cut???)
        self.simulate_constructed_rounds(4)
        self.simulate_draft_rounds(3)
        self.simulate_draft_rounds(3)
        self.simulate_constructed_rounds(4)

    def display_results(self):
        for player in self.players:
            print(f"{player.alias}: {player.get_tournament_history(self)}")
    
    def display_standings(self):
        for player in self.players:
            wins, losses = player.get_tournament_standing(self)
            print(f"{player}: {wins}-{losses}")
    
    def get_player_by_id(self, player_id):
        """
        Return a player object by its ID.
        """
        for player in self.players:
            if player.id == player_id:
                return player
        return None

    def get_top_n(self, n):
        """
        Return the top n players based on their score and common tiebreakers.
        """
        # First sort by the primary score based on wins and similarity score.
        primary_sorted_players = sorted(
            self.players, 
            key=lambda player: player.get_tournament_score(self), 
            reverse=True
        )

        # # RULES FOR TIEBREAKER. CANT FIX, TOO HARD
        # # Now sort by the common tiebreakers.
        # final_sorted_players = sorted(
        #     primary_sorted_players,
        #     key=lambda player: (
        #         player.get_tournament_score(self),  # Primary score
        #         player.get_opponents_match_win_percentage(self),  # OMW%
        #         player.get_game_win_percentage(self),  # GW%
        #         player.get_opponents_game_win_percentage(self)  # OGW%
        #     ),
        #     reverse=True
        # )

        return primary_sorted_players[:n]
    
    def get_win_prob_dataframe(self):
        """Retrieve the win probability for each player matrix as a DataFrame."""
        player_names = [str(player.id) + player.alias for player in self.players]
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(self.win_prob_matrix, index=player_names, columns=player_names)
        return df
    
    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}"


import pickle


import pandas as pd

# Load win_prob_matrix
player_win_prob_file = 'pvp_win_prob.csv'
# Read the CSV file into a DataFrame
player_win_prob_df = pd.read_csv(player_win_prob_file, index_col=0)  # Assuming the first column is the index

# Convert the DataFrame values to a NumPy array
tournament_win_prob_matrix = player_win_prob_df.values


from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cProfile
import pstats

def extract_tournament_data(tournament_id, tournament):
    top_64 = tournament.get_top_n(64)
    runs = {}
    for player in top_64:
        history = player.get_tournament_history(tournament)
        player_run = []
        for result in history:
            player_run.append((result[2], result[3]))
        runs[player.id] = player_run

    data = {
        "tournament_id": tournament_id,
        "top_64": [player.id for player in top_64],
        "top_64_runs": runs,
    }
    return data

# Number of tournaments to simulate
num_tournaments = 1

# Function to simulate a single tournament and extract data
def simulate_and_extract(tournament_id):
    with open("deck_manager.pkl", "rb") as f:
        deck_manager = pickle.load(f)
    with open("players.pkl", "rb") as f:
        players = pickle.load(f)
    # Create the tournament object inside this function
    tournament = Tournament(f"Tournament #{tournament_id}", players, tournament_win_prob_matrix)
    tournament.simulate_tournament()
    data = extract_tournament_data(tournament_id, tournament)
    return data

if __name__ == '__main__':
    # # Open the pickle file for writing
    # with open("E:/fab-data/simplified_tournament_data_new_pairings.pkl", "wb") as f:
    #     # Run the tournaments in parallel and extract data
    #     with ProcessPoolExecutor() as executor:
    #         # Generate tournament IDs and pass them to the executor
    #         tournament_ids = range(num_tournaments)
    #         for tournament_data in tqdm(executor.map(simulate_and_extract, tournament_ids), total=num_tournaments):
    #             # Serialize the extracted data to the file
    #             pickle.dump(tournament_data, f)
    #                 # Profiling the simulate_and_extract function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Call your simulate_and_extract function or the block of code you want to profile
    simulate_and_extract(0)  # Replace with a real tournament ID to test
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()