import random
import base64

# token pool is for utilizing github personal access token
class tokenPool:
  def __init__(self, tokens):    
    self.tokens = tokens.split(',')

  def get_token(self):
    return random.choice(self.tokens)