class HexModel(object):
    
    def predict(self, boards, **kwargs):
        """Given multiple boards predict the winner using as a 2-vector [black_win_prob, white_win_prob]"""
        raise  NotImplementedError()
        
    def fit(self, boards, winners, *args, **kwargs):
        """Given a list of boards and a winner of a game, train the model"""
        raise  NotImplementedError()
        
    def save(self, fileanem):
        """Serialize model to disk"""
        raise  NotImplementedError()
        
    def load(self, fileanem):
        """De-serialize model from disk"""
        raise  NotImplementedError()