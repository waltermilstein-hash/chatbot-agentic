"""
YAML file manager.
"""
import yaml
import os


class YamlFileManager:
    def __init__(self, file_path='/config/default/config.yml'):
        self.file_path = file_path
    def load(self):
        """Load the data from the YAML file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data

    def save(self, data):
        """Save the data to the YAML file."""
        with open(self.file_path, 'w') as file:
            yaml.safe_dump(data, file)

    def update(self, key, value):
        """Update a specific key in the YAML file."""
        data = self.load()
        data[key] = value
        self.save(data)

    def delete(self, key):
        """Delete a specific key from the YAML file."""
        data = self.load()
        if key in data:
            del data[key]
            self.save(data)
        else:
            raise KeyError(f"The key '{key}' does not exist in the YAML file.")

    def add(self, key, value):
        """Add a new key to the YAML file."""
        data = self.load()
        if key in data:
            raise KeyError(f"The key '{key}' already exists in the YAML file.")
        data[key] = value
        self.save(data)
