import random


class SecretSanta:
    def __init__(self, names):
        self.names = names
        self.restrictions = {
            'Amani': ['Fabienne', 'Jérome'],
            'Fabienne': ['Amani', 'Bassem'],
            'Jérome': ['Amani', 'Bassem', 'Nourzat'],
            'Bassem': ['Jérome', 'Fabienne'],
            'Nourzat': ['Jérome', 'Fabienne']
        }

    def is_valid_pair(self, giver, receiver):
        # Check if the receiver is not in the giver's restriction list
        return receiver not in self.restrictions.get(giver, [])

    def generate_pairs(self):
        # Make a copy of the names list to work with
        available_receivers = self.names[:]
        pairs = {}

        # Shuffle the list of names to ensure randomness
        random.shuffle(self.names)

        for giver in self.names:
            # Attempt to find a valid receiver who is not the giver and meets the restrictions
            valid_receivers = [r for r in available_receivers if r != giver and self.is_valid_pair(giver, r)]

            # If no valid receivers are left, reshuffle and restart the pairing process
            if not valid_receivers:
                return self.generate_pairs()

            # Select a random receiver from the valid list and assign the pair
            receiver = random.choice(valid_receivers)
            pairs[giver] = receiver

            # Remove the selected receiver from the list of available receivers
            available_receivers.remove(receiver)

        # Print the pairs line by line
        for giver, receiver in pairs.items():
            print(f"{giver} will give a gift to {receiver}")

        return pairs


# Example usage with the names provided
names = [
    "Zeina Tlaiss",
    "Aslam Mohamadaly",
    "Hugo Corbin",
    "Ryan Vernel",
    "Nourzat",
    "Amani",
    "Hanane",
    "Bassem",
    "Max",
    "Mathieu",
    "Naim",
    "Baptiste Mlynarz",
    "Stéphane",
    "Jérome",
    "Fabienne",
    "xavier"
]

# Initialize the SecretSanta class with the names
secret_santa = SecretSanta(names)

# Generate and print the Secret Santa pairs
secret_santa.generate_pairs()
