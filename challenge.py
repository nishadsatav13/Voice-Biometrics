import random

PHRASES = [
    "open the secure door",
    "my voice is my password",
    "authentication by voice",
    "unlock the system now",
    "voice biometrics login"
]

def get_random_phrase():
    return random.choice(PHRASES)

if __name__ == "__main__":
    print(get_random_phrase())
