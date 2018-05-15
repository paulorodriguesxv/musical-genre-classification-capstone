import os

def music_path(filename):
    return os.path.join(os.getcwd(), 'genres', filename)

def get_genre(path):
    _, genre = os.path.split(os.path.split(path)[0])
    return genre

def get_filename(path):
    _, filename = os.path.split(path)
    return filename