import numpy as np

class Callback():
    def __init__(self, enable_early_stop=False, enable_save_best_vloss=False, early_stop_delta=0.001, path_best_model='save_model/best_model.json') -> None:
        self.list_callback = {
            'early_stop': enable_early_stop,
            'save_best_vloss': enable_save_best_vloss
        }
        self.early_stop_delta = early_stop_delta
        self.last_vloss = None
        self.last_epoch = None
        self.current_epoch = 0
        self.best_vloss = None
        self.path_best_model = path_best_model

    def callback_manager(self, history, model):
        if not self.last_epoch:
            self.last_epoch = history['epoch']

        self.current_epoch = history['epoch']
        if self.list_callback['early_stop'] == True and self.early_stop(history=history) == True:
                print('Stop learning phase due to the early stop breakpoint')
                return True
        if self.list_callback['save_best_vloss'] == True:
            self.save_best_model(history=history, model=model)

        self.last_epoch = history['epoch']
        return False

    def early_stop(self, history) -> bool:
        if not self.last_vloss:
            self.last_vloss = float(history['val_loss_entropy'][self.current_epoch])
            return False
        
        if np.abs(self.last_vloss - float(history['val_loss_entropy'][self.current_epoch])) < self.early_stop_delta \
            and self.last_epoch != history['epoch']:
            return True
        
        self.last_vloss = float(history['val_loss_entropy'][self.current_epoch])
        return False
    
    def save_best_model(self, history, model) -> None:
        if not self.best_vloss:
            self.best_vloss = float(history['val_loss_entropy'][self.current_epoch])
            return

        if self.best_vloss >  float(history['val_loss_entropy'][self.current_epoch]) \
            and self.last_epoch != history['epoch']:
            #save the current best model and earse the previous
            model.save_weights(self, self.path_best_model)
            self.best_vloss = float(history['val_loss_entropy'][self.current_epoch])
        return



