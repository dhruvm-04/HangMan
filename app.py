import os
import random
import pickle
from collections import defaultdict
import numpy as np
import streamlit as st

# --- Configuration ---
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_WRONG_GUESSES = 6
CORPUS_FILE = 'corpus.txt'
TEST_FILE = 'test.txt'
ARTIFACT_DIR = 'artifacts'
ORACLE_PATH = os.path.join(ARTIFACT_DIR, 'oracle.pkl')
AGENT_PATH = os.path.join(ARTIFACT_DIR, 'agent.pkl')

# --- Default factories (picklable) ---
def one(): return 1
def alpha_len(): return len(ALPHABET)
def dd_one(): return defaultdict(one)
def zeros5(): return np.zeros(5)

# -----------------------------------------------
# Minimal environment (for play/eval)
# -----------------------------------------------
class HangmanEnv:
    def __init__(self, word_list, max_wrong_guesses=MAX_WRONG_GUESSES):
        self.word_list = word_list
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()

    def reset(self, specific_word=None):
        self.word = (specific_word if specific_word else random.choice(self.word_list)).upper()
        self.masked_word = "_" * len(self.word)
        self.guessed_letters = set()
        self.lives_left = self.max_wrong_guesses
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return (self.masked_word, self.guessed_letters, self.lives_left)

    def step(self, letter):
        letter = letter.upper()
        info = {"repeated": False, "wrong": False}
        if letter in self.guessed_letters:
            info["repeated"] = True
            return self._get_obs(), -2, self.done, info
        self.guessed_letters.add(letter)
        if letter in self.word:
            new_masked_word = list(self.masked_word)
            for i, char in enumerate(self.word):
                if char == letter:
                    new_masked_word[i] = letter
            self.masked_word = "".join(new_masked_word)
            if "_" not in self.masked_word:
                self.done = True
                return self._get_obs(), 50, self.done, info
            else:
                return self._get_obs(), 5, self.done, info
        else:
            self.lives_left -= 1
            info["wrong"] = True
            if self.lives_left <= 0:
                self.done = True
                return self._get_obs(), -50, self.done, info
            else:
                return self._get_obs(), -5, self.done, info

# -----------------------------------------------
# Oracle (inference only)
# -----------------------------------------------
class ProbabilisticOracle:
    def __init__(self):
        self.unigrams = defaultdict(one)
        self.total_unigrams = len(ALPHABET)
        self.bigrams = defaultdict(dd_one)
        self.bigram_totals = defaultdict(alpha_len)
        self.trigrams = defaultdict(dd_one)
        self.trigram_totals = defaultdict(alpha_len)
        self.positional_freq = defaultdict(dd_one)
        self.positional_totals = defaultdict(alpha_len)
        self.max_word_len = 0

    def get_letter_probabilities(self, masked_word, guessed_letters):
        scores = defaultdict(float)
        unguessed_letters = [l for l in ALPHABET if l not in guessed_letters]
        if not unguessed_letters:
            return {}
        padded_masked = f"^{masked_word}$"
        for letter in unguessed_letters:
            scores[letter] = np.log(self.unigrams[letter] / self.total_unigrams)
            for i, char in enumerate(masked_word):
                if char == "_":
                    p_i = i + 1
                    c_prev = padded_masked[p_i - 1]
                    c_next = padded_masked[p_i + 1]
                    c_prev2 = padded_masked[p_i - 2] if p_i > 1 else None
                    pos_score = self.positional_freq[i][letter] / self.positional_totals[i]
                    bi_score_1 = self.bigrams[c_prev][letter] / self.bigram_totals[c_prev]
                    bi_score_2 = self.bigrams[letter][c_next] / self.bigram_totals[letter]
                    tri_score = 1.0
                    if c_prev2:
                        tri_score = self.trigrams[c_prev2+c_prev][letter] / self.trigram_totals[c_prev2+c_prev]
                    scores[letter] += np.log(pos_score) + np.log(bi_score_1) + np.log(bi_score_2) + (np.log(tri_score) * 2.0)
        if not scores:
            return {}
        max_score = max(scores.values())
        exp_scores = {l: np.exp(s - max_score) for l, s in scores.items()}
        total_exp_score = sum(exp_scores.values())
        if total_exp_score == 0:
            return {l: 1.0 / len(unguessed_letters) for l in unguessed_letters}
        probs = {l: s / total_exp_score for l, s in exp_scores.items()}
        return probs

# -----------------------------------------------
# RL Agent (policy-only)
# -----------------------------------------------
class HangmanRLAgent:
    def __init__(self):
        self.q_table = defaultdict(zeros5)
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.01
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01
        self.vowels = "AEIOU"

    def _get_state(self, lives_left, masked_word):
        num_blanks = masked_word.count("_")
        if num_blanks == 1: blanks_state = 1
        elif num_blanks == 2: blanks_state = 2
        elif num_blanks == 3: blanks_state = 3
        elif num_blanks <= 5: blanks_state = 4
        else: blanks_state = 5
        return (lives_left, blanks_state)

    def _get_letter_from_action(self, action_idx, hmm_probs, guessed_letters):
        sorted_probs = sorted(hmm_probs.items(), key=lambda item: item[1], reverse=True)
        unguessed_sorted = [l for l, p in sorted_probs if l not in guessed_letters]
        if not unguessed_sorted:
            return None
        unguessed_vowels = [l for l in unguessed_sorted if l in self.vowels]
        unguessed_consonants = [l for l in unguessed_sorted if l not in self.vowels]
        letter_to_guess = None
        if action_idx == 0: letter_to_guess = unguessed_sorted[0]
        elif action_idx == 1: letter_to_guess = unguessed_sorted[1] if len(unguessed_sorted) > 1 else None
        elif action_idx == 2: letter_to_guess = unguessed_sorted[2] if len(unguessed_sorted) > 2 else None
        elif action_idx == 3: letter_to_guess = unguessed_vowels[0] if unguessed_vowels else None
        elif action_idx == 4: letter_to_guess = unguessed_consonants[0] if unguessed_consonants else None
        if letter_to_guess is None:
            letter_to_guess = unguessed_sorted[0]
        return letter_to_guess

    def choose_action(self, state, hmm_probs, guessed_letters, is_training=False):
        action_idx = int(np.argmax(self.q_table[state]))
        letter = self._get_letter_from_action(action_idx, hmm_probs, guessed_letters)
        return action_idx, letter

# -----------------------------------------------
# Deserialization (load artifacts)
# -----------------------------------------------
def deserialize_oracle(state: dict) -> ProbabilisticOracle:
    o = ProbabilisticOracle()
    o.unigrams = defaultdict(one, state['unigrams'])
    o.total_unigrams = state['total_unigrams']
    o.bigrams = defaultdict(dd_one, {k: defaultdict(one, v) for k, v in state['bigrams'].items()})
    o.bigram_totals = defaultdict(alpha_len, state['bigram_totals'])
    o.trigrams = defaultdict(dd_one, {k: defaultdict(one, v) for k, v in state['trigrams'].items()})
    o.trigram_totals = defaultdict(alpha_len, state['trigram_totals'])
    o.positional_freq = defaultdict(dd_one, {int(k): defaultdict(one, v) for k, v in state['positional_freq'].items()})
    o.positional_totals = defaultdict(alpha_len, {int(k): v for k, v in state['positional_totals'].items()})
    o.max_word_len = state['max_word_len']
    return o

def deserialize_agent(state: dict) -> HangmanRLAgent:
    a = HangmanRLAgent()
    a.q_table = defaultdict(zeros5, {eval(k) if isinstance(k, str) and k.startswith('(') else k: np.array(v) for k, v in state['q_table'].items()})
    a.lr = state['lr']
    a.gamma = state['gamma']
    a.epsilon = state['epsilon']
    a.epsilon_decay = state['epsilon_decay']
    a.min_epsilon = state['min_epsilon']
    a.vowels = state['vowels']
    return a

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(ORACLE_PATH) or not os.path.exists(AGENT_PATH):
        return None, None, "Artifacts not found. Please run script.ipynb to train and save artifacts."
    with open(ORACLE_PATH, 'rb') as f:
        oracle_state = pickle.load(f)
    with open(AGENT_PATH, 'rb') as f:
        agent_state = pickle.load(f)
    oracle = deserialize_oracle(oracle_state)
    agent = deserialize_agent(agent_state)
    return oracle, agent, None

@st.cache_data(show_spinner=False)
def load_words(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [line.strip().upper() for line in f if line.strip().isalpha()]

def evaluate_agent(agent, oracle, test_words, num_games=200):
    if not test_words:
        return None
    env = HangmanEnv(test_words)
    total_success = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    eval_words = random.sample(test_words, min(num_games, len(test_words)))
    for word in eval_words:
        obs = env.reset(specific_word=word)
        masked_word, guessed_letters, lives_left = obs
        done = False
        game_wrong_guesses = 0
        game_repeated_guesses = 0
        while not done:
            hmm_probs = oracle.get_letter_probabilities(masked_word, guessed_letters)
            state = agent._get_state(lives_left, masked_word)
            _, letter_to_guess = agent.choose_action(state, hmm_probs, guessed_letters, is_training=False)
            if letter_to_guess is None:
                break
            next_obs, _, done, info = env.step(letter_to_guess)
            if info["repeated"]: game_repeated_guesses += 1
            if info["wrong"]: game_wrong_guesses += 1
            masked_word, guessed_letters, lives_left = next_obs
        if "_" not in masked_word:
            total_success += 1
        total_wrong_guesses += game_wrong_guesses
        total_repeated_guesses += game_repeated_guesses
    success_rate = total_success / len(eval_words)
    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
    return {
        'games': len(eval_words),
        'wins': total_success,
        'success_rate': success_rate,
        'wrong_guesses': total_wrong_guesses,
        'repeated_guesses': total_repeated_guesses,
        'final_score': final_score
    }

# -----------------------------------------------
# UI
# -----------------------------------------------
st.set_page_config(page_title="Hangman RL App", page_icon="🎯", layout="centered")
st.title("Hangman RL — Streamlit App")

oracle, agent, err = load_artifacts()
test_words = load_words(TEST_FILE)

if err:
    st.warning(err)
    st.info("Make sure 'corpus.txt' and 'test.txt' exist and run script.ipynb to create artifacts.")
else:
    st.success("Artifacts loaded.")

tab1, tab2, tab3 = st.tabs(["Suggest Next Guess", "Play a Game", "Evaluate on Test Set"])

with tab1:
    st.subheader("Suggest the next letter")
    masked = st.text_input("Masked pattern (use _ for blanks, e.g., _A__E):", value="")
    guessed = st.text_input("Already guessed letters (e.g., AEIO):", value="")
    lives = st.slider("Lives left:", min_value=0, max_value=MAX_WRONG_GUESSES, value=MAX_WRONG_GUESSES)
    if st.button("Suggest"):
        if err:
            st.error(err)
        else:
            masked = masked.strip().upper()
            guessed_set = set([c for c in guessed.upper() if c in ALPHABET])
            if not masked or any((c != "_" and c not in ALPHABET) for c in masked):
                st.error("Enter a valid masked pattern (A-Z and _).")
            else:
                probs = oracle.get_letter_probabilities(masked, guessed_set)
                state = agent._get_state(lives, masked)
                _, letter = agent.choose_action(state, probs, guessed_set, is_training=False)
                if letter is None:
                    st.info("No letters left to guess.")
                else:
                    st.success(f"Suggested next letter: {letter}")
                    if probs:
                        top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                        st.write({k: round(v, 4) for k, v in top})

with tab2:
    st.subheader("Play against the trained agent")
    if "game" not in st.session_state:
        st.session_state.game = None

    secret = st.text_input("Secret word (letters only). Leave empty to random from test set:", value="")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Start/Reset"):
            if err:
                st.error(err)
            else:
                words = test_words if (not secret.strip()) else [secret.strip().upper()]
                if not words:
                    st.error("No test words found. Ensure test.txt exists.")
                else:
                    st.session_state.game = HangmanEnv(words)
                    st.session_state.game.reset(specific_word=words[0] if len(words) == 1 else None)
    with colB:
        if st.button("Next step"):
            if not st.session_state.game:
                st.error("Start a game first.")
            else:
                masked_word, guessed_letters, lives_left = st.session_state.game._get_obs()
                if st.session_state.game.done:
                    st.info("Game already finished. Reset to play again.")
                else:
                    probs = oracle.get_letter_probabilities(masked_word, guessed_letters)
                    state = agent._get_state(lives_left, masked_word)
                    _, letter = agent.choose_action(state, probs, guessed_letters, is_training=False)
                    if letter is None:
                        st.info("Agent has no letters left to guess.")
                    else:
                        st.session_state.game.step(letter)

    if st.session_state.game:
        mw, gl, ll = st.session_state.game._get_obs()
        st.write(f"Masked word: {mw}")
        st.write(f"Guessed letters: {''.join(sorted(gl))}")
        st.write(f"Lives left: {ll}")
        if st.session_state.game.done:
            if "_" not in mw:
                st.success("Agent won!")
            else:
                st.error(f"Agent lost. Word was: {st.session_state.game.word}")

with tab3:
    st.subheader("Evaluate on test set")
    n_games = st.number_input("Number of games", min_value=10, max_value=2000, value=200, step=10)
    if st.button("Run evaluation"):
        if err:
            st.error(err)
        else:
            if not test_words:
                st.error("No test words found. Ensure test.txt exists.")
            else:
                metrics = evaluate_agent(agent, oracle, test_words, num_games=int(n_games))
                if metrics is None:
                    st.error("Unable to evaluate (no test words).")
                else:
                    st.write({
                        "Games": metrics['games'],
                        "Wins": metrics['wins'],
                        "Success Rate %": round(metrics['success_rate'] * 100, 2),
                        "Total Wrong Guesses": metrics['wrong_guesses'],
                        "Total Repeated Guesses": metrics['repeated_guesses'],
                        "FINAL SCORE": round(metrics['final_score'], 2)
                    })
