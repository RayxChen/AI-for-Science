# from model_ensemble_invariance import FES
from model_attn_3 import FES
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils import get_data


if __name__ == "__main__":
    config = {
        'batch_size': 80,
        'lr': 4e-4,
    }

    epochs = 100
    name = "data_10K_attn_3_coord_dist_key"

    logger = TensorBoardLogger('logs', name=f'{name}')

    checkpoint_callback = ModelCheckpoint(
        filepath='./ckpt/{epoch:03d}' + f'_{name}',
        period=10,
        monitor='valid_loss',
        save_top_k=1,
    )
    model = FES(config)

    trainer = Trainer(gpus=-1, max_epochs=epochs,
                      logger=logger, checkpoint_callback=checkpoint_callback)
    # trainer = Trainer(gpus=-1, fast_dev_run=True)
    trainer.fit(model)

    _, _, test = get_data(batch_size=config['batch_size'])
    trainer.test(test_dataloaders=test)

